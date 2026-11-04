# C++ 静态规划的独立差分验收

`test/TaskflowMapping/cpp_differential.py` 保留文件名供现有工具导入；现为原生回归驱动，不内嵌 Python oracle，也不提供 `--prepare-only`。

```bash
python3 test/TaskflowMapping/cpp_differential.py \
  --tool build-core/tools/taskflow-opt/taskflow-opt \
  --output build/my-native-regression
```

它调用 C++ 规划器与检查器，保留合成图的明确预期、微型图闭式成本枚举、
22 个正向断言、39 个重新签名负例及 6 个 body/witness 检查。
测试中的哈希实现先通过未修改计划的往返检查；非法计划重新签名，避免仅因 checksum 过期被拒绝。
原生检查器通过不作为独立 Python 验证。v4 曾用 Python 参考完成 17 个变体、196 个完整计划差分；这是历史证据。v5 将参考代码快照保存后退出活动代码，不再运行第二套 Python 语义检查器。
完整构建、搜索、协议、缓存和 Amoeba 回归入口为 `scripts/run_taskflow_static_cpp.sh`。

## 负例与真实 body 检查

39 个重新签名负例包含既有 28 个检查，以及区域数、源操作、Taskflow 操作绑定、
实现与成本别名、外部输出消费者、初始冷热状态、计时边界、最终输出版本、
字节溢出和仅删除最终可见性同步等 11 个检查。

六个 body/witness 检查验证：有效 witness、保留旧 witness 修改实际算术、
修改 ReLU 零阈值、无 witness、无 witness 的非法算术，以及同时更新有效 body
和 witness 后重新分析形状和成本。最后一种使预计成本从 8 ms 变为 12 ms，
避免仅依靠旧元数据作出同一计划。这里的旧结构接口专用分析拒绝缺少 `orbit.plan` 的输入。v5 的 `GenericAnalysis` 是独立的通用入口，使用函数参数角色和 SSA 存储身份，无需完整 witness；两类入口的支持范围不能混淆。

## 历史证据与限制

开发验证目录为
`build/static-mapping-v4/cpp-migration/validation/differential-r3`：
19 个调用案例、22 个正向断言、39 个负例、6 个 body 检查全部通过。
最早的 `validation/differential` 失败记录保留；它暴露了 `i1 true` 完成值
在本地分析中被符号扩展为 -1 的问题，修复后重新运行成功。
统一复现脚本 `scripts/run_taskflow_static_cpp.sh` 使用当次最终构建再次执行
差分、原生搜索及既有 Core/Amoeba/search/cache 回归；应以该次输出为最终验收。

所有这里的成本都是静态估计。没有运行 NPU，也没有借助历史硬件结果证明
迁移性能。Python oracle 与 C++ 实现共享公开语义约定，因此还需保留非法计划
和实际 body 修改测试；差分一致本身不是完整形式化证明。未覆盖任意动态形状、
未知算术或任意精度转换。通用区域组覆盖和新的硬件执行优化不属于此迁移测试。
