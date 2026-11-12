# v32 AIR 标量 BF16 舍入修复

MLP2 的旧 compiled program 能完成三个调用，但没有满足原 RNE 数值参考。修复保留 source、reference 和容差，在 ORBIT 生成的 AIR 中局部实现 RNE；它不修改 vendor 或 core rounding register。新源码已通过 CPU 测试；r10 MLP2 的三个执行方案随后完成 324 calls 原 RNE 数值检查。whole-program NPU 对 phase-local NPU 为 3.990116×，CPU 仍更快；这不是 CP 搜索选出的加速结果。详见 [r10 上板报告](taskflow-npu-mlp-r10-v32.md)。

## 实测反例与定位

原 source：`/tmp/matrix-shared-mlp-r1/mlp16-layers2-whole-program/source.mlir`。输入及原参考：`/tmp/orbit-mlp-numerical-v32-r1/{inputs.npz,reference-2.npz}`；旧板上输出：[`outputs.npz`](/tmp/orbit-mlp2-whole-measure-v32-r1/outputs.npz)。source 每层先做 FP32 multiply/add，中间 ReLU 后 `arith.truncf f32→bf16`，最终输出 FP32。

独立顺序 FP32 计算和显式 BF16 RNE 逐元素等于原 reference。三个 actual outputs 每次均有 102/256 个元素与它不同，最大绝对误差 `0.0001220703125`，差值全部非正。把**诊断参考**的中间转换改成 RTZ 后，三个 actual outputs 均逐元素完全相等；中间 38/256 个 BF16 值的 RNE/RTZ 结果不同。

这定位了非 RNE 的中间转换，不能据此把原参考改成 RTZ。由于转换前 ReLU 非负，此例也无法区分 floor 和 toward-zero。安装的 `aie_kernels/linalg/mm_aie2p.h` 说明 core 的初始模式为 floor；安装的 `datamovement/affine_cast_f32_bf16.cc` 使用局部 conv_even 并恢复旧模式。实际旧 LLVM 保留 `fptrunc float→bfloat`，object 反汇编使用 `vconv.bf16.fp32`。这些证据支持环境 rounding mode 的定位，但不是任意负数输入或所有设备状态的实测证明。

独立诊断脚本和封存比较：`/tmp/audit_mlp2_rounding_v32.py`、[`summary.json`](/tmp/mlp2-rounding-audit-v32-r1/summary.json)、`/tmp/mlp2-rounding-audit-v32-r1/comparisons.npz`。这些是本地诊断证据路径，不是仓库测试的必需依赖。

## 精确修复边界

内部 helper [`ProgramGraphAIRBF16RNE.h`](../lib/TaskflowAIR/ProgramGraphAIRBF16RNE.h) 在 [`ProgramGraphAIR.cpp`](../lib/TaskflowAIR/ProgramGraphAIR.cpp) 的实际事件生成完成后、LowerAffinePass 前运行。它只处理 emitted clone 中默认或显式 nearest-even 的**标量 f32→bf16**；显式 RTZ 等其他模式、vector/tensor 和其他位宽保持原路径。原 source 与 selected graph 不变。

有限值的 unsigned bit 算法为 `high16(u + 0x7fff + ((u >> 16) & 1))`，integer add 不附加 overflow flags。它保留正负零，处理正负 ties、subnormal 和有限溢出到 infinity。NaN 使用独立分支保留符号及上部 payload，并置 quiet bit，避免变成 infinity；±Inf 保留。这里没有建模可观察的浮点异常 flag，也没有声称 nonfinite hardware 全流程已验收。

此实现不访问 rounding register，因此无需 save/restore，也不会改变邻近 mmul 的模式。安装 API 的 `set_rounding` 控制 accumulator→vector conversions；当前 accfloat→float 是同宽 cast，而 BF16 narrowing 和其他整数 SRS 可能观察全局模式。避免全 core 设置可以减少这些额外作用域。

## 验证与范围

独立 C++ 测试 [`program-graph-air-bf16-rne.cpp`](../test/TaskflowGeneric/program-graph-air-bf16-rne.cpp) 已编译、链接并通过，验证精确触发边界、source clone 隔离，以及**实际生成 IR** constant folding 的 16 个正负 zero/tie/subnormal/overflow/Inf/NaN 反例。CMake 提供纯 CPU target：

```sh
cmake --build BUILD --target check-taskflow-bf16-rne
```

该 target 不要求 AIR runtime 或 NPU。另一个独立 MLIR 原型通过解析、LLVM lowering 和本机 AIE2p CPU 编译；其 assembly 仅含整数 shift/add/bitwise/select，没有 BF16 conversion 或 crRnd 访问。数值邻点 oracle 检查 791680 个正负有限 f32 值，包括每个 BF16 bin 的 tie 邻域和 seeded random，零不一致；另检查八个 exceptional cases。该大样本证明针对原型算法，不能替代后续生产修改的测试。原型证据：`/tmp/explicit-bf16-rne-v32-test.json`。

新增 integer work 可能改变 inline code size、优化和执行成本。r10 已使用新 frozen compiler/contract 重新编译 MLP2，并保留**原 RNE reference 与原容差**完成上板复验；更长 MLP、其他转换模式及完整成本模型仍需分别验证。旧 numeric failure、原型 CPU PASS 和 r10 板上成功分别保留，不能互相替代。

## r10 通用 IR 兼容与实测

AIR 接受整数 RNE lowering 后，会给 `arith.trunci` 打印新版默认 property `overflowFlags = #arith.overflow<none>`；pinned MLIR 不支持该 property。[`ProgramRuntimeCompatibility.h`](../include/Taskflow/ProgramRuntimeCompatibility.h) 在 runtime generic IR 入口以独立未注册 Arith context 做结构化解析，仅清除这个精确的语义默认，再交原 registered parser/checker。普通 source parser 不变。nsw/nuw、额外 property、错误类型及普通 attribute 编码均拒绝；其他 operation 的 flags 和 metadata 保留。没有字符串泛化 strip，也没有修改 vendor。

receipt 绑定原 generic input、normalized 文本的实际 SHA-256 和移除数量，另保留最终 realized IR SHA。真实 MLP whole-program 清除两处默认属性；phase-local 四配置中仅 round 配置清除两处，其余为零。ON/OFF `check-taskflow-runtime-compatibility` 及真实 generic CLI/AIE verifier 均通过。

r10 frozen tool SHA 为 `0812d28cc6ca271a64e881a76d4f0edc9d0f35fec3b3d5312e308d3484baca03`。三方案各三个 fresh process、每 process 5 warmups+31 hot calls，共 324 calls 全部通过原 reference 检查。phase-local NPU 为 4 physical configurations，whole-program 为 1；实测改善属于这两个固定执行方案的比较。r9 的模型/测量保留原身份，不作为不同 r10 合同的已校准价格。
