# v7 复现与证据入口

从新 shell 运行，输出路径必须不存在：

```sh
cd /home/shiran/project/orbit
bash scripts/run_taskflow_program_mapping_v7.sh cpu /absolute/new/orbit-v7-cpu
bash scripts/run_taskflow_program_mapping_v7.sh all /absolute/new/orbit-v7-all
```

入口先构建和运行独立 Core、Amoeba、原搜索及直接 AIR 回归，再运行公共 mapping/整程序分片检查和两个原搜索实验。`all` 继续编译并执行四个归档候选、整程序尾片、非整除 GEMM、超容量加法→缩放、M5/M128 完整 FFN、exp/SiLU/SwiGLU 范围与边界探针。它是 fail-fast 入口；失败输出保留，不继续伪装成全部通过。参考本机 LLVM/AIR/AIE/XRT 安装路径沿用已有脚本；工具链缺失会明确失败。

M128 当前每次完整运行约9.7分钟；长验收使用三个独立 seed、每 seed 一次完整运行、零 warmup。其他普通验收为三 seed、各两次 warmup 与三次 hot。整个 `all` 需要比 M128 三次运行更长的时间。执行器使用既有 XRT runner；该 runner 来自已安装的后端包，但计算编译入口始终是原生生成的 AIR → aircc → AIE/ELF，没有调用 Triton 计算 recipe。

本次会话没有为重复跑完整 `all` 再次消耗所有长运行时间：完整 `cpu` 已作为一条命令通过；硬件各项依次实际执行，独立保留每项命令、编译、门禁、输出和时间。最终状态由 [本机结果报告](../build/direct-air-v7/RESULTS.md) 给出，不能只凭入口存在认定硬件已通过。

## 查证顺序

1. `build/direct-air-v7/program-search-r3/search/search.json`：父候选、动作链、拒绝、评价及排序；候选子目录保留真实 Taskflow、分析、公共计划及 AIR。
2. `build/direct-air-v7/program-selected-r3/cases.json`：实际交错实验与归档候选的绑定；不是另生成的一组执行参数。
3. `build/direct-air-v7/partition-m128-r1`：真实长验收使用的整程序 Taskflow、C++ 计划、16次调用、M8模板及原编译器身份。后续只增加会计/证明的修正通过独立等价性报告关联，不覆盖原目录。
4. `/home/shiran/project/aie-toy/results/orbit-direct-air-v7`：各硬件实验的 aircc 命令、后续 IR、ELF、工具链身份、逐阶段容量与 DMA 审计、原始输出及时间样本。
5. `build/direct-air-v7/full-regression-r2`：完整 CPU 回归的原始输出；后续局部修改的针对性测试在同级新目录保存。

M128 完整计时包含切片、runner staging/sync/submit/wait/readback、输出组装和逐调用证据写盘；不含 context 创建、完整输入/参考读取、初始完整输出分配、最终数值检查和完整输出序列化。context 与 aircc 时间单独记录。XRT wait 是主机观测的等待跨度，不是纯硬件 kernel 时钟；shim DMA 字节是生成程序的静态核对，不是 DDR 计数器测量。
