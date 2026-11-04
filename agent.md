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
