# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Tensormap and Ringbuffer Runtime build configuration
# All paths are relative to this file's directory (src/runtime/tensormap_and_ringbuffer/)
#
# This is a device-orchestration runtime where:
# - AICPU thread 3 runs the orchestrator (builds task graph on device)
# - AICPU threads 0/1/2 run schedulers (dispatch tasks to AICore)
# - AICore executes tasks via an aligned PTO2DispatchPayload + pre-built dispatch_args
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the Tensor constructor's validation logic).
#
# Compile-time knobs
# ------------------
# PTO2_PERF_LEVEL  (default 2 via pto_runtime2_types.h #ifndef)
#   Controls swimlane data-collection level, independent of PTO2_PROFILING.
#   0 = no collection, 1 = task only (version=1 JSON), 2 = task+phase (version=2 JSON)
#   Override via:  cmake -DPTO2_PERF_LEVEL=<0|1|2>  or  env PTO2_PERF_LEVEL=<0|1|2>
#
# PTO2_PROFILING   (default 1 via pto_runtime2_types.h #ifndef)
#   Controls scheduler cycle-stat logging and orchestrator sub-step statistics.
#   Independent of PTO2_PERF_LEVEL.

import os

_pto2_perf_level = int(os.environ.get("PTO2_PERF_LEVEL", "2"))

BUILD_CONFIG = {
    "aicore": {"include_dirs": ["runtime", "common"], "source_dirs": ["aicore", "orchestration"]},
    "aicpu": {"include_dirs": ["runtime", "common"], "source_dirs": ["aicpu", "runtime", "orchestration"]},
    "host": {"include_dirs": ["runtime", "common"], "source_dirs": ["host", "runtime", "orchestration"]},
    "orchestration": {"include_dirs": ["runtime", "orchestration", "common"], "source_dirs": ["orchestration"]},
    # cmake_defs: forwarded to cmake as -DKEY=VALUE for all targets.
    # Requires CUSTOM_COMPILE_DEFS support in the platform CMakeLists.txt
    # and corresponding gen_cmake_args() support in runtime_compiler.py.
    "cmake_defs": {
        "PTO2_PERF_LEVEL": _pto2_perf_level,
    },
}
