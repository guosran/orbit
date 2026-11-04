# ORBIT 项目背景

项目背景与架构定位已统一维护在 [agents.md](agents.md)。

## direct-air-v7 阶段记录（2026-11-04）

本轮以整个 program 的候选为中心：原搜索新增/组合 native 计算分块、访问分析发现的只读区域驻留与整程序独立轴分片，候选保留真实 MLIR、父候选和动作链。公共 mapping 明确区域、版本、副本、有效区间、使用者、资源与传输依赖；AIR 后端消费计划并接受最终 AIE 传输/容量审计。

研究定位见 [公共程序 mapping](docs/taskflow-program-mapping.md) 和 [Dato 对照](docs/taskflow-dato-overlap-v7.md)。语义事实（区域、共享、依赖、归约）、搜索选择（结构、分片、分组、驻留、调度）与后端机制（channel、DMA、FIFO、token）分开。Dato 的自动虚拟到物理映射与搜索能力被明确保留；借鉴其契约主要用于编译器生成的计划，不要求 ORBIT 用户先给出通信图。

已确认 M128 原单命令路径遭遇约60秒驱动 watchdog；独立阶段和 M 扩展诊断支持标量工作量过大的解释。完整 M16 通过两次 M8 程序命令在72.74秒得到正确输出。M128 三 seed 完整长验收现已全部通过：各完整 Y[128,2048] 由16次完整M8 NPU命令产生，约581.6秒/seed，原FP32容差未变。分块/驻留搜索已有8合法候选及1容量拒绝，readonly-panel 的实际 DMA 缺生产者问题已修复并通过小型完整程序真实执行。

最终状态、失败证据、测量边界与复现以 [本机 v7 报告](build/direct-air-v7/RESULTS.md) 为准；统一新目录入口是 [run_taskflow_program_mapping_v7.sh](https://github.com/guosran/orbit-artifact/blob/main/taskflow/scripts/run_taskflow_program_mapping_v7.sh)。独立审计边界见 [mapping review](docs/taskflow-program-mapping-review-v7.md)。

## 原 v7 冻结收尾（2026-11-04）

原队列全部完成：M128三seed完整输出、4候选60输出、整程序分片尾块15输出、M5/三维GEMM尾块/add→scale各15输出、exp/SiLU/SwiGLU各15输出均独立通过。exp边界保留原finite拒绝并另行分类。M128 package的native compiler为3e041…，后续冻结工具实际为2444…，各实际元数据和ELF/计划身份分别保留，未重写为单一版本。

冻结包位于 `/home/shiran/project/aie-toy/results/orbit-direct-air-v7/frozen-original-r1`，328文件完整性通过；最终结果见该结果根的 `validation-summary.json` 和本机RESULTS。原树后续仅报告更新。新开发仅在 `/home/shiran/project/orbit-data-v8`，原CGRA模型只读审计见新树 `docs/taskflow-cgra-model-boundary-v8.md`；没有训练新模型。NPU当前空闲，等待新树CPU验收完成后的明确调度。


## 阶段进展：公共数据组织与联合搜索 v8

补充任务已完成实现和验收：实际 Taskflow 分析驱动公共数据/排序边与共享副本，复用 Static memory/resource/link；phase-local 与 retain-across-users 接回原搜索，与 tiling 组合得到两轮容量耦合候选。公共副本39项、native mapping36项、Core/Amoeba/旧搜索与直接AIR回归、真实CGRA模型8个合约均通过。CGRA ML保留原II模型和生产scheduler duration，公共费用合约明确数据供给条件及通信归属，未重新训练。

新版本两个驻留候选在NPU上3seed共30完整输出通过；计划DMA由221184B降至122880B，热中位数18.658558/18.650719ms，差异不足以宣称加速。新M5完整FFN15输出通过；原M128完整FFN3seed长验收已全部通过，约581.6秒/完整program，原包/工具/ELF分别绑定。最后checker-only保护经完整CPU复跑，5候选与M5生成的语义产物逐字节等于已上机包，未将等价检查冒称额外NPU执行。

设计和范围见 `docs/taskflow-data-organization-v8.md`；验收、计时/模型边界和复现见 `docs/taskflow-data-organization-v8-validation.md`。入口 `../orbit-artifact/taskflow/scripts/run_taskflow_data_organization_v8.sh cpu|all NEW_DIRECTORY`。隔离构建和全部原始证据位于 `/home/shiran/project/orbit-data-v8/build/data-organization-v8`，旧v6/v7证据保持原状。有限whole-task共享方案先在CPU合成目标验证，AIR保留仍限单execution group内已证明的只读panel；任意跨stage SRAM共享、自动异构放置、Exact/profile公共provider仍是后续扩展。

用户已授权验收完成后提交并push；提交范围为相关源码、测试和文档，生成产物继续本地保留。


## Repository separation and queued FFN profiling

Core implementation, source semantics, mapping/search/checkers, backend lowering and compiler-only regression fixtures remain in this repository. Experiment orchestration, numerical input/reference generation, hardware execution/audits, model regression and validation reports now live in the sibling `guosran/orbit-artifact` repository under `taskflow/`. New build/output defaults are `orbit-artifact/builds/taskflow/{core,air,amoeba}` and `orbit-artifact/results/taskflow`; shared helpers accept ORBIT_SRC, ORBIT_BUILD_ROOT and ORBIT_RESULTS_ROOT. Core-only tests use standard-library test/Support and do not import artifact runtime code. See `docs/repository-layout.md` and `scripts/build-core.sh`.

Historic raw results, frozen tools and modified v8 source snapshots were physically moved under the companion's ignored local storage. Compatibility aliases preserve old absolute references; original manifests and measurements retain their bytes. Reproduce into fresh directories rather than rebuilding historical CMake caches. New evidence and code-generation tests verify the separation.

Queued next: evidence-first full FP32 FFN profiling. Decompose the original M128/16×M8 timeline, explain the N16 residency comparison, inspect current executed ELF/helper paths, perform controlled stage/full-graph measurements, then choose one or two supported optimizations and validate a frozen direct-AIR model on held-out candidates. Keep CGRA ML unchanged and preserve artifact/compiler versions. The main agent coordinates NPU use; current queued work starts with read-only evidence analysis while repository separation is validated.

## 阶段进展：整程序 profiling 与联合搜索反馈 v9

已完成原 M128 时间审计、完整 D2048/I11008 的 M1/M5/M8 分阶段与完整图实测、N16 驻留交错对照和实际执行 ELF 热点检查。原581.6秒由16次约36.35秒完整M8调用构成，记录的wait占完整wall约99.986%；三个同core串行GEMM解释主要时间，wait内计算/搬运/停顿仍为组合项。N16计划搬运减少44.4%未产生可证明加速，2/5个设备stage均仍是一次host提交，不能将静态字节量或额外stage直接当成实测DDR或已隔离配置成本。

公共候选新增可执行的 `scalar-product-vector-add-f32`、lanes8 实现选择，与原 tiling、只读副本驻留和两轮 native 搜索组合。实际body保持FP32产品、每输出K顺序和原1e-5 finite契约；initializer/math明确保留scalar。M1/M5/M8完整图三seed通过，交错热中位延迟下降7.37%/7.89%/6.83%。新版完整M128单seed profiling通过，541.8698秒、最大绝对误差2.14e-8；原版三seed长验收独立保留，没有冒称新版三seed。实际full M8两版分别绑定ELF43cf…/9ce0…，payload20480B、计划搬运306176000B均未增加或改变。

生产C++新增 native cost features、有限域 composite prediction 和 execution-fingerprint 精确实测反馈；原CGRA ML未重训。冻结scalar模型M3完整图留出误差0.474%，17×80尾块低估29–44%，原结果未改；新增逐source配置守卫，未标定/不匹配候选返回null。原搜索对N32两种驻留×两种实现进行真实四候选比较，完整144输出通过；原理论不能区分实现，实测反馈选择mixed8→Bpanel。原13合法proposal中4实测、7原生null、2未再访问，父链、checkpoint、缓存/模型/目标身份均有证据，不将静态重验当额外硬件执行。

已修复vector lowering前后循环遍历轴不一致，保留r2 timeout/错误AIR；lane4目标支持不足明确拒绝。独立SSA/K/lane检查、非整除K及超容量generic add→scale真实执行通过。另保留四input多phase后端stream次序失败并用窄目标能力守卫拒绝；未声称通用四stream修复。Core/Amoeba/搜索/AIR31组、C++成本47项、真实CGRA模型8合约及相关checkpoint/fitter/编排检查通过，独立Core不链接AIR/AIE/XRT/Neura。

报告与复现位于伴随仓库 [FFN profiling v9](https://github.com/guosran/orbit-artifact/blob/main/taskflow/docs/taskflow-ffn-profiling-v9.md)。源码仍以整个program的结构、资源、调度和数据组织候选为中心；此次NPU实现/驻留实验为公共搜索提供可执行策略、适用域和完整command成本反馈。没有全FFN设备trace、硬件DDR计数或任意多core预测；single-core trace仅用于其有界窗口。复现入口支持新目录的CPU、训练/冻结/留出及优化分阶段；原始大产物继续保存在伴随仓库的本机ignored results/builds，reviewable证据单独发布。

## Latest stage: target-driven predictive search v10

The completed v10 stage is recorded in English in [agents.md](agents.md) and
[the validation record](docs/taskflow-implementation-search-v10-validation.md).
It adds target-provided memory architecture and implementation-aware prediction
to original whole-program search, and validates an automatically selected new
execution with 474 total independent device output checks. The frozen model's
six-holdout MAPE is 1.6304%; full M5 configuration transfer is correct but shows
no speedup. Historical M128 results retain their earlier identities. Core and
experiment artifacts remain in their separate repositories.
