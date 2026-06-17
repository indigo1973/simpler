/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * @file l2_swimlane_collector_aicore.h
 * @brief AICore performance data collection interface
 *
 * Provides lightweight performance recording interface for AICore kernels.
 * Uses dcci for efficient cache management instead of memory barriers.
 */

#ifndef PLATFORM_AICORE_L2_SWIMLANE_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_L2_SWIMLANE_COLLECTOR_AICORE_H_

#include "common/l2_swimlane_profiling.h"
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"  // L2SW-PROBE2: get_aicore_profiling_flag / get_l2_swimlane_aicore_head (remove with probe)
#include "common/platform_config.h"  // L2SW-PROBE2: GET_PROFILING_FLAG + probe bits

// Include platform-specific timestamp implementation
// Build system selects the correct inner_kernel.h based on platform:
// - src/a2a3/platform/onboard/aicore/inner_kernel.h (real hardware)
// - src/a2a3/platform/sim/aicore/inner_kernel.h (simulation)
// Both provide unified get_sys_cnt_aicore() interface
#include "inner_kernel.h"

// ============= Public Interface =============

/**
 * AICore-local rotation state. Tracks which buffer this core is currently
 * writing into and the next slot. Rotation is detected by the per-task
 * `cur_buf_ptr` (delivered via the dispatch payload) changing vs `cached_buf`,
 * so AICore never reads a shared AICPU-written cache line.
 */
struct L2SwimlaneAicoreLocalState {
    __gm__ L2SwimlaneAicoreTaskBuffer *cached_buf = nullptr;
    uint32_t slot_within_buf = 0;
};

/**
 * Record task execution performance data.
 *
 * AICore writes a slim L2SwimlaneAicoreTaskRecord into its current per-core
 * L2SwimlaneAicoreTaskBuffer at `records[slot_within_buf++]`. AICPU owns buffer
 * rotation (it enqueues a full buffer to the ready queue and pops a fresh one
 * from the free_queue every PLATFORM_AICORE_BUFFER_SIZE dispatches), giving
 * unbounded per-core records via host recycling. The current buffer pointer is
 * delivered to AICore **through the per-task dispatch payload**
 * (`PTO2DispatchPayload::l2_swimlane_cur_buf_ptr`, stamped by the scheduler
 * dispatch path right after the AICPU rotation hook). AICore reads it out of
 * the payload it already `dcci`s every task — it does NOT poll a shared
 * `L2SwimlaneActiveHead` line. (The earlier shared-head poll wedged the a5
 * AIC->AICPU FIN handshake: reading a GM line AICPU concurrently writes stalls
 * the a5 AICore pipeline so it never signals FIN. a2a3 tolerated it; a5 does
 * not.) AICore detects rotation simply by the payload pointer differing from
 * its cached copy and resets the slot.
 *
 * AICPU and AICore never read each other's hot-path data. The host
 * post-processor joins the AICore stream (multi-buffer per core, in order)
 * with the AICPU stream by `reg_task_id` at flush time. See
 * `docs/dfx/l2-swimlane-profiling.md`.
 *
 * Ordering: the dispatch path stamps `cur_buf_ptr` after the rotation hook and
 * before the `wmb()` + `write_reg(DATA_MAIN_BASE)`, so the value AICore reads
 * matches the buffer AICPU will recycle. The completion-before-dispatch
 * invariant (AICore per core is single-threaded; AICPU does not dispatch task
 * K+1 until K FIN'd) guarantees all prior records were written and dcci'd out
 * before AICPU enqueues the old buffer.
 *
 * @param cur_buf_ptr     Current per-core AICore record buffer (GM device ptr)
 *                        for THIS dispatch, from the payload. 0 when AICPU had
 *                        no free buffer this batch (record dropped; AICPU
 *                        already bumped dropped_record_count).
 * @param local           Per-core AICore-local state (caller-owned static)
 * @param task_token_raw  Full task identity (PTO2 encoding for tensormap_and_ringbuffer
 *                        runtime: `(ring_id << 32) | local_id`; plain task index
 *                        zero-extended for host_build_graph). The caller in the
 *                        ringbuffer runtime reads this from
 *                        `exec_payload->local_context.async_ctx.task_token.raw`
 *                        which is already in AICore cache (it was just dcci'd for
 *                        the kernel call), so no extra GM load.
 * @param reg_task_id     Per-core dispatch token (low 32 bits of the per-core
 *                        monotonic dispatch_seq). Per-dispatch unique within
 *                        a core; serves as the host-side join key against the
 *                        AICPU record stream. Required because SPMD with
 *                        `block_num > num_cores` (and MIX cluster spread)
 *                        dispatch the same `task_token_raw` multiple times to
 *                        the same core — each dispatch needs its own AICore
 *                        record matched to its own AICPU record, which
 *                        task_token_raw alone cannot disambiguate.
 * @param start_time      Start timestamp (get_sys_cnt)
 * @param end_time        End timestamp
 */
__aicore__ __attribute__((always_inline)) static inline void l2_swimlane_aicore_record_task(
    uint64_t cur_buf_ptr, L2SwimlaneAicoreLocalState *local, uint64_t task_token_raw, uint32_t reg_task_id,
    uint64_t start_time, uint64_t end_time
) {
    if (cur_buf_ptr == 0) {
        // AICPU couldn't pop a fresh buffer from free_queue this batch. Drop
        // silently — AICPU side already bumped dropped_record_count.
        return;
    }
    __gm__ L2SwimlaneAicoreTaskBuffer *buf = reinterpret_cast<__gm__ L2SwimlaneAicoreTaskBuffer *>(cur_buf_ptr);
    // Detect rotation by the payload-delivered buffer pointer changing — no
    // cross-core read of an AICPU-written channel. AICPU rotates exactly every
    // PLATFORM_AICORE_BUFFER_SIZE dispatches, in lockstep with this per-core
    // slot counter, so the slot stays in [0, BUFFER_SIZE).
    if (buf != local->cached_buf) {
        local->cached_buf = buf;
        local->slot_within_buf = 0;
    }

    uint32_t slot = local->slot_within_buf;
    if (slot >= PLATFORM_AICORE_BUFFER_SIZE) {
        // Defensive: AICPU should rotate before this can happen. If it
        // didn't, refuse to write past the end rather than corrupt adjacent
        // memory.
        return;
    }

    __gm__ L2SwimlaneAicoreTaskRecord *record = &buf->records[slot];
    record->start_time = start_time;
    record->end_time = end_time;
    record->task_token_raw = task_token_raw;
    record->reg_task_id = reg_task_id;
    local->slot_within_buf = slot + 1;

    // Flush record to GM so host can read it after the buffer is enqueued.
    // The completion-before-dispatch invariant guarantees this dcci has hit
    // GM before AICPU enqueues the buffer.
    dcci(record, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);

    // L2SW-PROBE2: extra per-task cross-core read to isolate the a5 stall cause.
    // Built on the fixed path (buffer ptr came from payload, no head read above),
    // so reaching here proves AICore progressed normally. Remove after the probe.
    uint32_t l2sw_probe_flag = get_aicore_profiling_flag();
    if (GET_PROFILING_FLAG(l2sw_probe_flag, PROFILING_FLAG_L2SW_PROBE_READ_OWN)) {
        // Read back AICore's OWN just-written record line (profiling region,
        // AICore-written). If this stalls -> reading the profiling region itself
        // is toxic on a5. If it completes -> the head stall is about reading a
        // cross-agent (AICPU-written) line, not the region.
        dcci(record, SINGLE_CACHE_LINE);
        volatile uint64_t probe_own = record->start_time;
        (void)probe_own;
    }
    if (GET_PROFILING_FLAG(l2sw_probe_flag, PROFILING_FLAG_L2SW_PROBE_READ_HEAD)) {
        // Positive control: read the shared head line (profiling region,
        // AICPU-written) — the exact toxic op from #983. Should reproduce 507000.
        __gm__ L2SwimlaneActiveHead *probe_head = get_l2_swimlane_aicore_head();
        if (probe_head != nullptr) {
            dcci(probe_head, SINGLE_CACHE_LINE);
            volatile uint64_t probe_h = probe_head->current_buf_ptr;
            (void)probe_h;
        }
    }
}

#endif  // PLATFORM_AICORE_L2_SWIMLANE_COLLECTOR_AICORE_H_
