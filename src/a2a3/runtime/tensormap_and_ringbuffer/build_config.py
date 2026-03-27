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
# Each target supports an optional "extra_defines" list to inject compile-time
# preprocessor macros without modifying pto_runtime2_types.h. Examples:
#
#   "extra_defines": ["PTO2_ORCH_PROFILING=1", "PTO2_SCHED_PROFILING=1"]
#
# These become -DNAME=VALUE flags for AICPU and AICore compilation
# (they do NOT apply to the host target).

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "orchestration"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicpu", "runtime", "orchestration"]
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "runtime", "orchestration"]
    },
    "orchestration": {
        "include_dirs": ["runtime", "orchestration"],
        "source_dirs": ["orchestration"]
    }
}
