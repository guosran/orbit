# 公共 Taskflow → Amoeba / Neura

本次交付保持公共候选的语义身份，将目标资源选择留在 Neura adapter。公共
`candidate.mlir` 和 `plan.json` 不被目标 lowering 改写。每份 `feedback.json`
保留原 `candidate_id`、`source_id`、plan SHA-256、materialized MLIR SHA-256。

## 实际结果

| 验证 | 结果 | 证据类型 |
|---|---|---|
| 14 个小尺寸公共候选，每个 3 个种子 | 42/42 数值通过，最大绝对误差 1.20e-7 | 旧 host lowering → LLVM shared library → CPU 执行 |
| Neura kernel construction | 13/14 成功 | 编译 IR；尚不等于映射 |
| 公共 GEMM baseline | 原 mapper II `[1, 3]`；整图 scheduler makespan **6653 cycles** | 两个目标 stage 的真实 mapper 结果与 production scheduler trace |
| 同源公共 GEMM M/N 分块 | 原 mapper II `[1, 3, 1, 3, 1, 3, 1, 3]`；整图 **6662 cycles** | 八个目标 stage 的 mapper 与 scheduler trace |
| 旧回归 + 新 adapter lit | **35/35**（34 旧 + 1 新） | verifier、结构变换、mapper、scheduler、trace、cache |
| 篡改公共 MLIR body 而保留 witness | 拒绝 | adapter 认证负例 |

没有实体 CGRA 执行。CPU wall time 仅是 correctness replay 的附带记录；它不是
CGRA 时间。6653/6662 是 mapper 驱动的调度周期估计，也不是硬件测量。该固定
资源方案下 M/N 分块没有收益，不据此宣称性能提升。

证据根目录为 `build/backend-decoupling/`。公共结果在
`<case>/<candidate>/amoeba/feedback.json`，完整 mapper IR 在
`gemm/{base,mn}/amoeba/mapped-scheduled.mlir`，资源清单在同目录
`mapper-candidates.jsonl`，mapper/调度日志与完整命令在 `mapping.json`。
旧回归证据在 `amoeba/regression.json`、`amoeba/lit.log`、
`amoeba/legacy-mapped-scheduled.mlir`、`amoeba/legacy-mapper-cache.log`。

## 边界与实现

1. `taskflow-opt` 先从公共 MLIR 重新导出 plan，验证 `orbit.plan` witness
   与真实 task body、切片、依赖一致；adapter 再与传入 JSON 全量比较。
2. `prepare-structured-taskflow-for-neura` 将同一 task 内的顺序顶层 affine
   loop nests 拆成目标实现 stage。例如 fill 一次，然后依次执行 K chunks。
   stage 之间使用实际 `done_writes → will_writes` 状态形成依赖，并保留原始
   storage operands、输入完成依赖、原 task 名与 stage 序号。源候选文件不变。
3. 拆分仅支持 constants + affine nests、单输出 storage、无跨 loop SSA
   结果和无 read-completion 结果。它不猜测更通用 fission 的合法性。
4. 原 `construct-hyperblock-from-task`、counter classification、Taskflow→Neura
   pass 继续生成并认证 source iteration domain。任何旧证明失败仍然拒绝。
5. 可选 `--map` 使用原完整 Neura dataflow preparation，然后原资源枚举器、
   materializer、mapper 和 production scheduler。资源策略限定每个目标 stage
   最多一个 CGRA。实际 PE 矩形由指定架构推导，trip count 由 source certificate
   推导，adapter 不手工填入这些字段。此实验架构为 4×4 CGRA，每 CGRA 4×4 PE。
6. 资源层 `candidate-0` 是该公共候选对应的第一个目标资源方案；它没有覆盖
   外层公共 candidate hash。计分 provenance 为
   `source-owned-all-unit-mapped-duration`，单位 cycles；真实 mapper II 也以
   cycles 表示。未运行 mapper 的候选保持 unscored。没有将 NPU 毫秒写入旧字段。
7. legacy verifier 通过注册的 `OutputCoordinateProofProvider` 调用原 Neura
   `analyzeReplicaOutputCoordinates`，保留原认证检查。独立 core 未注册该服务
   时对需要它的旧输入 fail closed。

旧 host lowering 支持新增公共计算使用的纯 `math` 运算，因此小 FFN 的
`math.exp` 能按标准 SiLU 重放。数值参考是原始未变换 source plan 上的 NumPy
计算，不是候选循环的另一份逐指令解释。输入包含负数、零和非对称权重；检查
全输出、形状和有限值，FP32 使用 `atol=rtol=2e-5`，同时保存绝对误差。

## 支持矩阵

| 公共候选 | CPU replay | Neura kernel construction | 本次实际 mapper/scheduler |
|---|---|---|---|
| GEMM baseline / M-N / 顺序 K | 支持，3 seeds 全通过 | 支持 | baseline 与 M-N 已通过；K 未尝试 |
| GEMV baseline / M-N / 顺序 K | 支持，3 seeds 全通过 | 支持 | 未尝试 |
| GEMM→ReLU baseline / M-N / K / fission | 支持，3 seeds 全通过 | 支持 | 未尝试 |
| GEMM→ReLU fused 多输出 task | 支持，3 seeds 全通过 | **unsupported**：目标 stage adapter 只支持单输出 | unsupported |
| 小标准 SwiGLU FFN baseline / M-N / K | 支持，3 seeds 全通过 | 支持，包括尚留待后续 lowering 的 math ops | 未尝试；kernel construction 不证明可映射 |
| Qwen 完整 FFN | 本次没有在 Amoeba replay | 未尝试 | 未尝试；NPU 验收见另文 |

首次将尚含 `arith.constant` 的 kernel-construction IR 直接送 mapper 时，原检查
明确报 `mapper body requires prepared Neura dataflow; unsupported residual
operation arith.constant`。随后补齐原完整 preparation pipeline，通过 mapping。
失败与后续成功命令均保留在 `gemm/base/amoeba/mapping-attempt.json`，没有删除
检查或修改诊断来绕过限制。

## 干净 shell 复现

先按 [公共后端说明](taskflow-backends.md) 构建独立 core 并生成 manifest。
LLVM 必须为 `6146a88f60492b520a36f8f8f3231e15f3cc6082`。此机器 LLVM 的
C++ 标准库 ABI 对应 `/home/shiran/bin/g++`（GCC 15.2）；应使用同一编译器。

```sh
cd /home/shiran/project/orbit
export LLVM_BUILD=/home/shiran/project/llvm-project/build
export PATH="$LLVM_BUILD/bin:$PATH"
cmake --fresh -S . -B build-amoeba -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/home/shiran/bin/g++ \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DNEURA_ENABLE_PYTHON_BINDINGS=OFF \
  -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so \
  -DZLIB_INCLUDE_DIR=/usr/include
ninja -C build-amoeba tools/mlir-amoeba-opt/mlir-amoeba-opt -j3
ninja -C "$LLVM_BUILD" not
"$LLVM_BUILD/bin/llvm-lit" -sv -j2 build-amoeba/test
python3 -m venv build-amoeba/.venv
build-amoeba/.venv/bin/pip install numpy==2.5.4
build-amoeba/.venv/bin/python scripts/taskflow_amoeba.py \
  --plan build/backend-decoupling/gemm/base/plan.json \
  --mlir build/backend-decoupling/gemm/base/candidate.mlir \
  --source-plan build/backend-decoupling/gemm/source.json \
  --output build/backend-decoupling/gemm/base/amoeba --map
```

将上例的 `gemm/base` 改成 `gemm/mn` 可复现同源 M/N 候选。省略 `--map`
则只做 kernel construction 与 CPU replay。`--core-tool`、`--optimizer`、
`--llvm-build`、`--architecture` 可覆盖工具与架构路径。默认三个种子为 0、1、2，
一次 warmup 后保留三个完整图同步 CPU wall-time 样本；该计时不参与 Neura 评分。

`NEURA_ENABLE_PYTHON_BINDINGS` 默认仍为 ON，原 Python 构建行为不变；C++ 工具
可关闭该选项，以使用本机没有 MLIR Python bindings 的 pinned LLVM build。

## 35 项 lit 清单

以下清单来自本次成功运行的 `llvm-lit --show-tests build-amoeba/test`。

- `TaskflowAmoeba/structured-stages.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-ray-carried-reduction-unproven.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-ray-carried-reduction.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-rebase-disjoint-history.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-independent-recurrences.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-multi-output.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-recurrence-cut.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-reject-output-alias.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-reject-output-ownership.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-source-captured-domain.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-source-reject-alias.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-source-reject-effect.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-source-reject-stale-domain.mlir`
- `multi-cgra/taskflow/joint-scheduling/fission-task-source.mlir`
- `multi-cgra/taskflow/joint-scheduling/in-place-mn-tiling-payload-contract.mlir`
- `multi-cgra/taskflow/joint-scheduling/in-place-mn-tiling-proof.mlir`
- `multi-cgra/taskflow/joint-scheduling/joint-neighborhood-action-replay.mlir`
- `multi-cgra/taskflow/joint-scheduling/map-joint-scheduling-all-unit-cache.mlir`
- `multi-cgra/taskflow/joint-scheduling/map-joint-scheduling-all-unit-illegal-shape.mlir`
- `multi-cgra/taskflow/joint-scheduling/map-joint-scheduling-all-unit-unproved-domain.mlir`
- `multi-cgra/taskflow/joint-scheduling/memory-feedback-sequential-k-live-results.mlir`
- `multi-cgra/taskflow/joint-scheduling/neura-joint-rewrite-sibling-indexed-load-dedup.mlir`
- `multi-cgra/taskflow/joint-scheduling/neura-joint-rewrite-sibling-load-dedup.mlir`
- `multi-cgra/taskflow/joint-scheduling/neura-store-indexed-roundtrip.mlir`
- `multi-cgra/taskflow/joint-scheduling/producer-consumer-composite-source-domain.mlir`
- `multi-cgra/taskflow/joint-scheduling/prove-static-active-transfer-shapes-casts.mlir`
- `multi-cgra/taskflow/joint-scheduling/prove-static-active-transfer-shapes-conservative.mlir`
- `multi-cgra/taskflow/joint-scheduling/prove-static-active-transfer-shapes-mixed-join.mlir`
- `multi-cgra/taskflow/joint-scheduling/prove-static-active-transfer-shapes.mlir`
- `multi-cgra/taskflow/joint-scheduling/ready-task-placement.mlir`
- `multi-cgra/taskflow/joint-scheduling/sibling-fusion-source-domain.mlir`
- `multi-cgra/taskflow/joint-scheduling/source-iteration-domain-capture.mlir`
- `multi-cgra/taskflow/joint-scheduling/source-iteration-domain-forged-proof.mlir`
- `multi-cgra/taskflow/joint-scheduling/source-iteration-domain-partition-lineage.mlir`
- `multi-cgra/taskflow/joint-scheduling/static-active-transfer-payload.mlir`
