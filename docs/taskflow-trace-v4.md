# v4 稀疏标记硬件 trace

`scripts/taskflow_npu_trace_v4.py` 使用既有 MultiLaunchRunner/XRT 路径，
在本次编译的临时 PATH 中提供 aiecc adapter。AIR 转换完成后，adapter 保存原始
`npu.source.mlir`，在本地产物中插桩，再调用安装好的真实 aiecc；没有修改安装包。
`--prepare` 只编译，不打开设备。正常执行自行获取 `/tmp/npu.lock`，禁止外套同一锁。

v3 的问题包括：订阅 EVENT0/1 却没有实际发射标记；向量、锁和端口事件很密集；
有限 trace 窗口不能证明完整性。最终 v4 在每组两个输入 acquire 完成后发射
`aie.event(0)`，在对应输入 release 前发射 `aie.event(1)`。标记包围一次
K-block MAC；不包含等待输入、输出初始化及全图 CPU 节点。

订阅 EVENT0、EVENT1、USER_EVENT0 和 USER_EVENT1，其他槽为 NONE。
原广播 15 继续启动 trace。输出 DMA await/free 后向每个被跟踪 core 发射
USER_EVENT0 表示范围结束，然后发射有限的 64 个 USER_EVENT1 排空 padding，
最后由 shim USER_EVENT2 广播 14 停止 trace。padding 明确排除在 MAC 区间之外。
trace BD 使用 64 字节 AXI burst，与最小 IRON 稀疏采集路径一致。
运行返回后才同步并读取输出 BO。停止配置不自动证明采集完整；审计必须观察到
每个被跟踪 core 的全部预期 marker 对和随后 USER_EVENT0 范围结束事件。

AIR 当前仅为八个 core 中的六个配置 trace packet flow（row 3–5）。插桩覆盖
八个 core，但捕获完整性只对明确列出的六个 core 判断。不能因此声称全设备、
完整 FFN 或 CPU SwiGLU 被 trace 覆盖。

## 复现

```bash
cd /home/shiran/project/orbit
source /home/shiran/project/aie-toy/scripts/env.sh
python scripts/taskflow_aie_trace_minimal_v4.py \
  --output-dir /home/shiran/project/aie-toy/results/orbit-static-v4-hardware/minimal-16k \
  --trace-bytes 16384
python scripts/taskflow_npu_trace_v4.py --case gemm --iterations 3 \
  --trace-bytes 65536 \
  --output-dir /home/shiran/project/aie-toy/results/orbit-static-v4-hardware/markers-64k
```

使用新输出目录，保存失败，不覆盖旧实验。最小程序以两种容量复跑；ORBIT GEMM
确认标记后再执行 `--case ffn_m5` 和 `--case ffn_m128`，以独立目录重复采集。
`--case all` 可以顺序运行三者。trace off/on 使用相同输入、权重、算术和精度，
热完整图样本至少三次；结果继续保存 CPU 节点、实际 start/wait/sync、全部输出。

## 独立审计

```bash
python test/TaskflowMapping/trace_v4.py --selftest
python test/TaskflowMapping/trace_v4.py \
  --root /home/shiran/project/aie-toy/results/orbit-static-v4-hardware/kmarkers-final512k \
  --output build/static-mapping-v4/validation/trace-final512k-audit-replay.json
```

对 `kmarkers-final1m`、`minimal-offon16k`、`minimal-offon32k` 分别重复上述审计，
使用不同输出文件。若审计整个硬件结果根目录，保留的前两轮失败也会被检测到，
因此整目录总体检查应失败；不能删除这些失败以制造全部尝试均成功的印象。

审计重新解码 uint32 原始数据，检查保存的 cycles 与 Perfetto 事件来源一致，
并检查实际物理 MLIR 中确有标记。对 ORBIT 128×256 物理输出分块，预期每个
被跟踪 core 有 `(M/128)*(N/256)*(K/64)` 对标记；不能根据观察到的数量倒推预期数。
最小程序预期四次 scale 调用。缺 marker、缺范围结束、顺序错误、core 缺失都不通过。
即使 marker 范围完整，`full_graph_complete` 和 `compute_only_calibration_eligible`
仍为 false。cycles 没有伪装成微秒，没有推测 host/device 时钟对齐。

开发阶段 GEMM 稀疏标记的 CPU 编译已成功，原始适配错误与修正后的产物分别保留于
`build/static-mapping-v4/trace-compile-marker-only` 和 `trace-compile-marker-only-r2`。
硬件成功和完整性以新结果及独立审计为准，不以编译成功代替。

最小 16 KiB / 32 KiB 采集均数值正确，各有 16 个非零原始 word；独立复解码
得到四对预期标记。首轮 ORBIT `markers-64k` 数值全部通过但 raw 全零，保留失败。
该轮使用 AIR 隐式 burst 和逐核直接停止；后续 `markers-burst64` 使用上述显式
burst 与广播停止。两项修改是对齐现有稀疏采集路径，不能在没有实测时断言是哪项
解决问题。`trace-first-audit-corrected.json` 将无 manifest 的全零捕获列为失败。

第三轮 `kmarkers-drain64` 的 GEMM 独立审计通过：六个被跟踪 core 各一对有序
MAC 标记，所有范围结束事件可见，全部 off/on 输出通过。最终真实 FFN 每核预期
1,376 对标记；AIR 按八列分割缓冲，故最终比较使用 512 KiB 与 1 MiB 两种容量。
这项容量预算与真实标记、事件精简和有限排空一起使用，不是单纯扩大旧高频窗口。
初版标记位于输出 acquire 后，会早于 trace 启动且可在下一工作循环多发标记；
第二轮虽 raw 非空仍被审计拒绝，证据保留在 `markers-burst64`。

## 最终实测与审计

512 KiB 和 1 MiB 的三种图全部数值正确；连同两种最小程序容量，16 份 raw 捕获
均通过独立标记范围完整性检查，共 99,092 对预期标记。完整图 60 份输出和最小
程序 4 份 off/on 输出通过，所有同条件 off/on 输出逐元素相同。审计汇总：
`build/static-mapping-v4/validation/trace-final-audit.json`。

| trace 容量 | 图 | off 完整图中位 ms | on 完整图中位 ms | on/off |
|---|---|---:|---:|---:|
| 512k | ffn_m128 | 14.577739 | 14.186599 | 0.9732 |
| 512k | ffn_m5 | 10.827152 | 12.411168 | 1.1463 |
| 512k | gemm | 0.402731 | 0.619566 | 1.5384 |
| 1m | ffn_m128 | 13.898173 | 13.886353 | 0.9991 |
| 1m | ffn_m5 | 10.129867 | 10.429620 | 1.0296 |
| 1m | gemm | 0.404141 | 0.810222 | 2.0048 |

每个条件包含两次排除的执行和三次完整热样本。未锁频，顺序测量；小差异不能视为
加速。此表只描述同一新版驱动中插桩、BO 扩展、同步读取、布局改变及有限排空
的总扰动，不与历史版本数字作加速比较。FFN 仍每图三次真实 start/wait，
CPU SwiGLU 明确保留。

可复现最终两配置采集（替换输出目录为不存在的新目录）：

```bash
python scripts/taskflow_npu_trace_v4.py --case all --iterations 3 \
  --trace-bytes 524288 --output-dir /tmp/orbit-trace-repeat512k
python scripts/taskflow_npu_trace_v4.py --case all --iterations 3 \
  --trace-bytes 1048576 --output-dir /tmp/orbit-trace-repeat1m
```

结构搜索所选 N=256 分块候选的独立硬件审计入口为
`test/TaskflowMapping/structural_hardware_v4.py`，其报告位于
`build/static-mapping-v4/validation/structural-n256-hardware-audit.json`。
它独立回放源程序生成 float64 参考，验证七份完整输出、三次实际提交/等待、
源操作 exact cover、依赖顺序和文件身份。预计 1.636019 ms、实测中位 1.806159 ms，
相对误差 -9.42%；计时为热 host-to-host 区域边界，不是设备纯计算时间。
