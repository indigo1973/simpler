# tensormap_and_ringbuffer：泳道与 PROFILING 三轴说明

本目录为 a2a3 生产用 PTO2 运行时。与性能相关的控制分为**三个相互独立**的维度（编译宏与运行时开关），避免把「调度日志 / ORCH 统计」与「泳道 JSON 采集」绑死在一起。

## 1. 三轴一览

| 轴 | 形式 | 作用 |
|----|------|------|
| `PTO2_PROFILING` 及其子宏（`PTO2_ORCH_PROFILING`、`PTO2_SCHED_PROFILING`、`PTO2_TENSORMAP_PROFILING`） | 编译期 | 调度循环 LAP、DEV_ALWAYS 汇总日志、ORCH 周期统计等；层级 `#error` 依赖见 `pto_runtime2_types.h` |
| `PTO2_PERF_LEVEL`（0 / 1 / 2） | 编译期 | 泳道相关代码编译进二进制的**粒度**；**不**与 `PTO2_PROFILING` 族互斥 |
| `Runtime::enable_profiling` / `--enable-profiling` | 运行时 | 是否做 perf 初始化、设备侧写缓冲、Host 采集与 `perf_swimlane_*.json` 落盘 |

关系简述：

- 关 `--enable-profiling`：即使编译了泳道代码，也不应在热路径上执行**仅为泳道**的 `get_sys_cnt*` 与写缓冲（由 `enable_profiling` 门控）。
- 关 `PTO2_PROFILING`：仍可单独开 `PTO2_PERF_LEVEL` 做泳道；调度器汇总日志等会消失。
- `PTO2_PERF_LEVEL` 只影响「编译进哪些泳道逻辑」，不替代 `--enable-profiling` 作为落盘开关。

## 2. `PTO2_PERF_LEVEL` 档位

定义见 [`runtime/pto_runtime2_types.h`](runtime/pto_runtime2_types.h)。

| 值 | 泳道语义（设备侧） | 导出 JSON `version`（由 Host 按 header 与数据决定） |
|----|-------------------|--------------------------------------------------------|
| 0 | 最轻：不调用 `perf_aicpu_complete_record`；AICore 写起止时间与 reg token 后，由 AICPU 在 completion 路径上提交该条 `PerfRecord`（递增 `PerfBuffer::count`，否则 flush/Host 侧会认为无数据） | 0：`tasks` 中省略 `dispatch_time_us` / `finish_time_us`，`fanout` 为空 |
| 1 | 含 AICPU task 元数据（dispatch/finish、fanout 等） | 1：无 phase 段 |
| 2 | 在 1 基础上含 sched / orch phase | 2：含 phase 相关字段 |

设备在 `perf_aicpu_init_profiling` 时把当前编译的 level 写入 `PerfDataHeader::swimlane_format_level`；Host 在 `export_swimlane_json` 中读取并选择字段集。若实际采集到 phase 数据，会把 JSON `version` 提升到 2。

## 3. 构建：如何设置 `PTO2_PERF_LEVEL`

**唯一来源**：在 [`runtime/pto_runtime2_types.h`](runtime/pto_runtime2_types.h) 里修改 `#ifndef PTO2_PERF_LEVEL` 下的默认值（`0` / `1` / `2`），然后**全量重编** aicore、aicpu、host。工程侧**不再**通过 CMake 或环境变量注入 `-DPTO2_PERF_LEVEL`，避免覆盖头文件里的设定。

同一次运行中三目标必须来自同一次修改后的头文件，否则 `PerfDataHeader::swimlane_format_level` 与二进制内联的泳道逻辑会不一致。

## 4. 主要改动文件索引（实现参考）

| 区域 | 文件 |
|------|------|
| Level 宏定义 | `runtime/pto_runtime2_types.h` |
| `enable_profiling` 字段条件 | `runtime/pto_orchestrator.h` |
| ORCH 泳道写与 `PTO2_PERF_PHASE` 分支 | `runtime/pto_orchestrator.cpp` |
| 调度 / 泳道拆分 | `aicpu/aicpu_executor.cpp` |
| AICore 计时与 `enable_profiling` 同分支 | `aicore/aicore_executor.cpp` |
| Perf 共享头与 AICPU 写 level | `src/a2a3/platform/include/common/perf_profiling.h`、`platform/src/aicpu/performance_collector_aicpu.cpp` |
| JSON 导出 | `platform/include/host/performance_collector.h`、`platform/src/host/performance_collector.cpp` |
| 编译入口 | `src/a2a3/platform/onboard/{aicore,aicpu,host}/CMakeLists.txt`、`sim/.../CMakeLists.txt`（`PTO2_PERF_LEVEL` 仅由头文件决定） |

## 5. 与旧文档的差异

历史文档里常见「`--enable-profiling` 依赖 `PTO2_PROFILING=1`」等表述；以本 README 与当前源码为准：`PTO2_PROFILING` 管日志与统计宏，泳道编译粒度由 `PTO2_PERF_LEVEL` 管，落盘由 `enable_profiling` 管。
