# a2a3 tensormap_and_ringbuffer：性能采集与泳道图输出

> **范围**：仅 **`src/a2a3/runtime/tensormap_and_ringbuffer/`** 及与之配套的 **a2a3 platform**（如 `perf_profiling.h`、`device_runner`、`performance_collector`）中与 perf/泳道相关的衔接。其它 runtime（host_build_graph、aicpu_build_graph、a5 等）不在此方案内。  
> **详细说明**：见 [`docs/profiling_levels.md`](src/a2a3/runtime/tensormap_and_ringbuffer/docs/profiling_levels.md)。

---

## 设计原则（两轴分离）

| 维度 | 作用 |
|------|------|
| **编译宏 `PTO2_PERF_LEVEL`** | 控制**是否编译进**哪一档 **Host–Device 共享内存 perf**（任务级 / 任务+阶段扩展 / 完全关闭）。**不**随单次运行切换。合法取值 **0、1、2**（在 `common/perf_profiling.h` 中定义并 `#error` 校验）。 |
| **`Runtime::enable_profiling` + CLI `--enable-profiling`** | 控制**本次运行**是否在 **LEVEL>0** 的前提下启用设备写缓冲、主机采集与导出；仍为 **bool**，不改为整型 level。 |

**宏定义位置**：`PTO2_PERF_LEVEL` 放在 **`src/a2a3/platform/include/common/perf_profiling.h`**（与 `PerfRecord`、共享内存布局同源），便于 **Host**（`performance_collector`、`device_runner`）与 **Device** 共用同一编译开关，而无需 Host 依赖 `pto_runtime2_types.h`。  
`tensormap` 的 **`pto_runtime2_types.h`** 通过 **`#include "common/perf_profiling.h"`** 引入该宏，避免重复定义。

### `PTO2_PERF_LEVEL` 分档（与 `PTO2_PROFILING=1` 联用）

| 值 | 共享内存任务 perf | Phase / orch 扩展、`core_to_thread` 等 | 主机 perf 区与泳道导出 |
|----|-------------------|----------------------------------------|-------------------------|
| **0** | 不编/不使用 | 无 | `device_runner` 不因 `enable_profiling` 分配 perf；`export_swimlane_json` 编译期禁用路径 |
| **1** | 有（`enable_profiling` 时） | 无 | 有任务级数据；JSON 通常**无** `core_to_thread`（设备不写） |
| **2** | 有 | 有（`enable_profiling` 时） | 含 phase、orch summary、`core_to_thread` 等，便于泳道 v2 风格与 **Scheduler DISPATCH → Core** 的 flow 映射 |

说明：

- `enable_profiling == false`：不 `perf_aicpu_init_*`、不写 PerfBuffer/PhaseBuffer；主机不采集导出（与 `PTO2_PERF_LEVEL` 无关，但若 **LEVEL=0** 主机侧本身也不会起 perf 管线）。
- **`LEVEL=0` 且 `enable_profiling==true`**：设备侧 **`profiling_enabled` 恒为 false**（`enable_profiling && (LEVEL>0)`）；编排器里 **`orch->enable_profiling`** 同样与 **`LEVEL>0`** 与运算后下发，避免 orchestrator 仍按「已开 profiling」走扩展路径。

### 与 `PTO2_PROFILING` 的关系

- **`PTO2_PROFILING=0`**：大量 AICPU 调度/编排侧 profiling 代码**不编译**；且 **`PTO2_ORCH_PROFILING` / `PTO2_SCHED_PROFILING` 必须为 0**（头文件约束）。  
- **`PTO2_PERF_LEVEL`** 的语义在 **`PTO2_PROFILING=1`** 下才有意义；`PROFILING=0` 时不讨论 level。  
- **AICore `aicore_executor.cpp`（tensormap）** 未使用 `PTO2_PROFILING`，任务级计时由 **`PTO2_PERF_LEVEL` + `enable_profiling`** 控制。

### 日志中的时间与仪器开销

- **`sched_start` / `sched_end` / `sched_cost`**：墙钟区间**包含**循环内打点、perf 写入等（若开启）；**刻意在**打印首条 sched 汇总 **`DEV_ALWAYS` 之前**取 `sched_end_ts`，**不包含**后续大段日志格式化时间。  
- **日志里的 `orch_cost`（`orch_end_ts - orch_cycle_start`）**：**宽于**纯 `orch_func`，可包含 ORCH 分项日志、`PERF_LEVEL>=2` 写共享内存、reassign 等直至取 `orch_end_ts` 前的逻辑。

---

## 泳道 JSON：`core_to_thread`

- **含义**：`core_to_thread[core_id] = sched_thread_idx`（`-1` 未分配）；由 **`perf_aicpu_write_core_assignments`** 在编排结束后写入（**`PTO2_PERF_LEVEL >= 2` 且 `enable_profiling`**）。  
- **在泳道图中的作用**：`tools/swimlane_converter.py` 用它确定 **哪条 Scheduler 线程泳道** 上的 **DISPATCH** 与 **哪个 Core** 上的任务条画 **flow**；缺失时用时间投票启发式推断。  
- **何时没有该字段**：见 `docs/profiling_levels.md`（如 `LEVEL<2`、未开 profiling、phase header magic 不匹配、`collect_phase_data` 未读到 `num_cores>0` 等）。

### `run_example.py` 与环境变量

- 若构建为 **`PTO2_PERF_LEVEL=0`**，请在运行前设置 **`export PTO2_PERF_LEVEL=0`**（与编译一致），脚本会**跳过** `swimlane_converter` 调用，避免无输入仍跑转换。未设置时默认按 **`"1"`** 尝试生成泳道（需与**实际编译**的 level 一致）。

---

## 实现要点（检查清单）

### 编译宏

- `PTO2_PERF_LEVEL`：**`perf_profiling.h`** 默认以仓库当前定义为准（可用 `-DPTO2_PERF_LEVEL=` 覆盖）；**0/1/2** 外触发 `#error`。  
- **`#if PTO2_PERF_LEVEL >= 2`**：phase init、`perf_aicpu_record_phase`、`perf_aicpu_write_orch_summary`、`perf_aicpu_write_core_assignments`、`perf_aicpu_flush_phase_buffers`、orchestrator `CYCLE_COUNT_LAP_RECORD`（在 `PERF>=2 && enable_profiling` 时）等。  
- **`#if PTO2_PERF_LEVEL > 0`**：任务级 PerfBuffer、`get_sys_cnt` 类热路径打点（与 `ORCH_PROFILING`/`SCHED_PROFILING` 的 wait_cycle 等配合时另见头文件）。  
- **调度/编排日志墙钟**（`sched_*_ts`、`orch_*_ts`）：在 **`PTO2_PROFILING=1`** 下可取 **`get_sys_cnt`**，**不**随 `PTO2_PERF_LEVEL==0` 置 0，便于仍打印耗时（与共享内存 perf 分离）。

### 运行时标志

- `runtime.h`：`bool enable_profiling`（**LEVEL=0** 时主机不启 perf 区；设备不写共享内存 perf）。  
- **AICore**：`profiling_enabled = runtime->enable_profiling && (PTO2_PERF_LEVEL > 0)`。  
- **AICPU**：同上；`phase_profiling_enabled = runtime->enable_profiling && (PTO2_PERF_LEVEL >= 2)`。  
- **`PTO2OrchestratorState::enable_profiling`**：创建 runtime 时 **`runtime->enable_profiling && (PTO2_PERF_LEVEL > 0)`** 写入各 `orchestrators[i]`。

### 未采用的方向

- ~~全仓库将 `enable_profiling` 改为 `int perf_level`~~  
- ~~CLI `--enable-profiling` 带 level 1/2~~（level 由 **重编译** `-DPTO2_PERF_LEVEL=` 决定；脚本侧用 **`PTO2_PERF_LEVEL` 环境变量** 与 level 0 对齐行为。）

---

## 验证建议

```bash
# 默认若未改宏：LEVEL 可能为 0，需显式打开任务级 perf 再导出泳道
CXXFLAGS="-DPTO2_PERF_LEVEL=1" CC=gcc CXX=g++ pip install .
# 或 build/cache 重建后：
# CXXFLAGS="-DPTO2_PERF_LEVEL=2" ...   # 含 phase / core_to_thread

python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/paged_attention/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/paged_attention/golden.py \
  -p a2a3 -d 0 --enable-profiling

# 构建为 LEVEL=0 时：与编译一致，避免脚本仍跑 swimlane_converter
export PTO2_PERF_LEVEL=0
python examples/scripts/run_example.py ... --enable-profiling

# 仿真
python examples/scripts/run_example.py \
  -k examples/a2a3/tensormap_and_ringbuffer/matmul/kernels \
  -g examples/a2a3/tensormap_and_ringbuffer/matmul/golden.py \
  -p a2a3sim --enable-profiling
```

换编译器或清理 **`build/cache`** 后再 `pip install .`，避免 CMake 缓存指向已删除的临时路径。

---

## 相关文件（速查）

| 区域 | 路径 |
|------|------|
| `PTO2_PERF_LEVEL` 定义 | `src/a2a3/platform/include/common/perf_profiling.h` |
| Runtime2 类型（include 上述头） | `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h` |
| `enable_profiling` | `runtime/runtime.h` |
| Host：perf 门控 | `platform/sim/host/device_runner.cpp`、`platform/onboard/host/device_runner.cpp` |
| Host：导出 JSON | `platform/src/host/performance_collector.cpp` |
| 泳道合并 | `tools/swimlane_converter.py` |
