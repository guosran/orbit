# v32 异构区域成本的校准与独立审查

本轮工具从同一 typed assignment 的真实 C++ 特征和已封存执行 receipt 建立局部 phase 模型。这里提供的是可复核的方法与 CPU 测试；r3/r4 的静态诊断和失败记录保留原身份。后续实测分别见 [r9 校准报告](taskflow-npu-calibration-r9-v32.md) 和 [r10 MLP2 报告](taskflow-npu-mlp-r10-v32.md)，不同合同的 timing 与价格不能直接混用。

## 原生特征与身份

[`make-heterogeneous-feature-config.py`](../scripts/make-heterogeneous-feature-config.py) 为当前 BF16 matrix+relu source 创建 public AllSearch 的显式 singleton 域，使用已有 materialize/checker/backend/cost callback 导出 `edge-features.json`。这属于 receipt 提取，不是优化效果实验。原 source 两个 task 先标 lineage，再分发 initializer，与现有三个 task 的 preparation 对齐。示例命令：

```sh
python3 scripts/make-heterogeneous-feature-config.py \
  --base PREP/config-npu-2.json --output NEW-config.json \
  --tasks 2 --storages 4 --device npu --tile 32 --distribute
FROZEN_TOOL PREP/gemm64.mlir --search --backend=air \
  --max-rounds=1 --max-candidates=1 \
  --target-config=NEW-config.json --search-output=NEW-search
```

没有模型时成本可以 unknown，原生特征仍会导出。提取后必须将 parent graph 和有序 child graph IDs 与**实际 compiled manifest** 全部比较；不同 witness 的特征不得借用。r3 的 16/32/64/96 CPU/NPU、48 CPU、48 tile16 NPU 共十项均已逐项匹配。96/48 后续编译问题意味着这些静态匹配不构成它们的硬件成功证据。新合同必须重新导出与比较。

helper 保留 `--tile 32` 的等轴兼容用法，并支持 `--tiles 16 32 48` 分别指定 M/N/K。两种参数互斥；非正 extent 或不足三个轴均拒绝。prep6/r4 预备设计为：NPU train16/tile16、32/tile32、32/tile16、64/tile32、64/tile16×16×64、96/tile16×32×48；CPU train16/32/64/96 各一个设计；heldout48 CPU 和 NPU tile16×16×48。它们是待新合同及实际编译确认的设计，不能由 helper 成功生成配置推断后端或硬件支持。

`mechanism-facts.v13` 不能重建全部 v32 特征：它缺少实现分项 work、完整 scratch ABI 和多配置绑定数量。GEMM64 NPU 的真实 C++ 特征是 4 BO、5 configurations、20 次参数绑定、49152 B TO-sync、16384 B FROM-sync。不能仅依据公共参数数量计价，也不能把 v13 schema 改名为原生特征。索引工具只用 v13 交叉检查 graph 和 application input-reset/output-copy bytes。

## 固定策略与索引封存

[`build-heterogeneous-calibration-index.py`](../scripts/build-heterogeneous-calibration-index.py) 接受显式 case list，每项包含 `id`、`split`、`shape`、`ranking_group`、`measurement`、`features`、`mechanism_facts`。路径相对 spec 文件解析。当前配置面向 train 16/32/64/96 与 heldout48；未来更换设计必须先冻结新 spec。每个 fresh process 一项，CPU/NPU 同源同输入使用相同 ranking group。

```sh
python3 scripts/build-heterogeneous-calibration-index.py preregister \
  --spec CASES.json --output NEW-registration.json
python3 scripts/build-heterogeneous-calibration-index.py build \
  --spec CASES.json --registration NEW-registration.json --output NEW-index
python3 scripts/calibrate-heterogeneous.py fit \
  --index NEW-index/index.json --output NEW-model
python3 scripts/calibrate-heterogeneous.py validate \
  --index NEW-index/index.json --model-dir NEW-model --output NEW-validation
```

### 同形状的显式执行设计划分

默认不指定 `split_policy` 时，保留原 train16/32/64/96、holdout48 规则。MLP 的相同 shape、不同 tile/retention/grouping 等方案可显式使用 `split_policy: "explicit-case-design-v1"`；每 case 仍指定 `split: "train"` 或 `"holdout"`，且两组均非空，`shape` 必须是任意正整数，不能是 bool 或浮点数。

每 case 增加完整静态设计声明：

```json
{
  "design": {
    "source": {"path": "SOURCE.mlir", "sha256": "ACTUAL_SOURCE_SHA256"},
    "parent_graph_id": "ACTUAL_PARENT_GRAPH_ID",
    "region_graph_ids": ["ACTUAL_CHILD_GRAPH_ID_IN_REGION_ORDER"]
  }
}
```

其余 `id`、`ranking_group`、`measurement`、`features`、`mechanism_facts` 与旧格式相同。`source.path` 相对 spec 解析，SHA 必须匹配真实字节；parent 和有序 child IDs 必须匹配 native C++ features，不能用 case 名称或自定义 plan label 替代。source→graph 的算术/实现证明仍由原生 compiler receipt 提供；Python 不新增一份编译器或 source 等价性判定。

预注册只读取 source 与静态 features，不打开 measurement labels；registration 中的 `case_designs` 记录规范化设计、`design_sha256` 和静态文件的 path/SHA。设计 hash 绑定 source SHA、parent/有序 child IDs 和声明 shape，不含 process ID 或测量路径。原 spec 本身也计算 SHA，因此其余声明同样被冻结。

跨 split 出现相同 `(parent_graph_id, ordered region_graph_ids)` 一律拒绝，即使更换 case id、shape 标签、source 路径或仅增加 source 注释。同一 split 内的 fresh-process 重复允许，保持相同设计身份。相同源程序、相同 shape 但不同合法 child graph 的执行方案可以分属 train/holdout；不同文件名本身不能创建独立 holdout。设计独立性和 rank 仍根据真实 witness/feature rows 判断，不能只数 hash 或 process 个数。

build 会再次校验 source/静态 features 与冻结 registration，并把设计 hash 写入每个 index case；编译器、manifest、graph、ABI 和 process 的原精确检查保持。calibrator、固定 phase features 和 accuracy/ranking gate 没有放松，也没有按完整设计 hash 查价格。新的 grouped/retention 方案仍须实际导出、编译、完成数值检查，不能由登记成功推断后端支持。

索引构建验证 identity→compiled→manifest SHA、exact parent/ordered child witness、配置数、配置数×完整 ABI 参数数，以及 measurement-Python process identity。样本文件只计算 SHA，不解析 timing labels。真正数值断言、完整编译/ELF/proof 链及 exclusive phase sum 仍由已有 calibrator 和独立数值审计验收。

默认固定策略每 phase 两项：CPU body 为 call+source operations；reset/copy 为 call+bytes；control 为 call+configuration count；NPU wait 为 configuration count+source operations；submit 为 configuration count+configuration argument bindings；sync 为 count+bytes。它只覆盖当前固定 whole-module 实现组合与单区域 dispatch 结构，不能代替任意 task 实现混合、区域数量或 ABI 变化的校准。没有完整 plan-ID 价格表。缺项、外来 family/contract 和越界特征保持 unknown。

现有 calibrator 冻结整个 index SHA，包括 heldout 引用的 SHA。因此它不支持模型冻结后原地补入新 heldout 哈希。可以先独立隔离采集 heldout，由索引工具只封存字节，然后只拟合 train，模型冻结后才打开 heldout labels。若要求物理采集也必须晚于模型冻结，需要新版本支持“冻结 heldout 声明 + append-only evidence seal”；不能改写旧 frozen index。

## 可辨识性与联合支持范围

[`audit-heterogeneous-feature-design.py`](../scripts/audit-heterogeneous-feature-design.py) 只读显式 case 的 sealed native features，计算设计 rank、归一化 condition number、各项 range 与联合凸包支持。它不读取 timing，不增设验收硬门槛：

```sh
PYTHON_WITH_NUMPY_SCIPY scripts/audit-heterogeneous-feature-design.py \
  --spec FEATURE_CASES.json --output NEW-design-audit.json
```

r3 静态诊断如下。完整捕获结果保存在 [`heterogeneous-calibration-design-audit-v32.json`](../test/TaskflowGeneric/fixtures/heterogeneous-calibration-design-audit-v32.json)，测试输入保存在 [`heterogeneous-feature-design-v32.json`](../test/TaskflowGeneric/fixtures/heterogeneous-feature-design-v32.json)。两者都不含 timing labels。

- submit 两列 rank=1：全部 NPU ABI 均为四个 BO，bindings 恒等于 4×configuration count，不能独立辨识两个系数。
- wait 两列 rank=2，但归一化 condition number≈14.84；其余两列设计约 2.82–3.16。
- 48 tile16 的 `(configuration_count, source_operations)=(12,444672)` 逐项在 train range 内，却在四个 tile32 training designs 的联合凸包外。96 覆盖 count=12，不等于训练了小 work、多 launch 的组合。

因此该 heldout 是未见 tile/work 组合的泛化 challenge，不能称严格插值已经覆盖。可以在查看 heldout timing 前预注册一个合法的新 training design，或保持 challenge 并如实接受 gate fail；不能看到误差后调参数。独立 feature rows 取决于不同 source/tiling 设计，不能只按 shape 数量计数；同 shape 不同 tile 可能增加 rank，也可能仍共线。重复 process 能改善噪声估计，不能增加设计 rank。NNLS 的共线系数只用于预测，不解释成已分别测出的 pure launch、setup 或 DMA 参数。

## 环境、时钟与验收

在采集前冻结随机顺序/seed、输入/reference SHA、warm/hot counts、timeout、设备锁及 quiet-window 规则。三 fresh Python processes 应按 round 随机轮换 case；不要根据已见标签挑最快窗口。保留所有尝试、失败和被排除的 pilot。外部 wrapper 记录实际 measurement-Python PID/exec lineage、command、affinity、wall/monotonic timestamps、源码及工具 SHA、退出码和锁使用情况。

每个 process 前后，在计时外记录 loadavg、CPU stat、CPU/memory/I/O pressure、可用内存、可读的 governor/frequency 和温度，并标明不可用项。记录同时发生的编译或其他设备用户。这些快照描述观察到的条件，不能证明全程空闲或频率恒定；验证后的快照也不能补充成历史校准证据。

主机登记的 HX370 CPU0–3 最大频率为 5.15 GHz，CPU4–11 为 3.29 GHz。未固定 affinity 的旧测量可能受跨核迁移影响，但这些最大值不是测量时实际频率，也不足以单独解释全部历史波动。新 wrapper 将测量进程固定在 CPU0，并封存 affinity/policy 文件，不更改系统策略或其他进程。固定核仍不能消除同核竞争、温度及动态频率的影响。

先取各 process 的 hot-call median，再聚合 process medians，完整保留逐调用和逐进程表。验收沿用已有门槛：每个完整 typed witness 至少三个有证据的 fresh processes，全部组覆盖，以及组相对误差中位数≤20%。Top1 只在 process-median min/max 区间分离且模型无 tie 时判 pass/fail，否则 unknown。凸包诊断没有变成新的硬 gate。

完整 call 边界仍是 reset + exclusive regions + control + output copy。NPU wait 联合包含 setup/body/teardown/完成等待，不能重复另加分析价格。静态 proof、上板数值、模型 accuracy/ranking、CP-SAT 相对固定基线的硬件 speedup 必须分别报告。

## CPU 回归

```sh
python3 test/TaskflowGeneric/heterogeneous_calibration_index_v32.py
PYTHON_WITH_NUMPY_SCIPY test/TaskflowGeneric/heterogeneous_feature_design_v32.py
```

两项独立测试程序均通过：index/config 测试现共 **11 例全部通过**，覆盖旧默认与新 explicit 协议的端到端 intake、同 shape 不同 child、同 split process 重复、跨 split 设计 alias 拒绝、source SHA/正整数 shape/缺失设计/未知协议负例，以及既有等轴兼容、独立 M/N/K 与冲突参数。样本可故意为非 JSON 字节，验证登记不读取 labels；原 graph/compiled 精确核验仍保留。另一个 design 测试验证 submit rank 缺口、单项 range 不足以证明联合支持、以及特征哈希变动拒绝。测试只依赖仓库内 fixture 和临时生成的 synthetic receipts，不依赖外部 `/tmp` 产物，不执行 NPU。
