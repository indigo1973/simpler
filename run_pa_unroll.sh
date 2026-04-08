#!/usr/bin/env bash
# Paged-attention unroll: build, one warmup, then timed rounds without profiling
# (default 10) and with --enable-profiling (default 10). Average elapsed (us) is parsed
# from device logs, same rules as
# tools/benchmark_rounds.sh (PTO2_PROFILING sched/orch timestamp lines).
# Swimlane JSON conversion is disabled; use swimlane_converter.py manually if needed.
#
# Optional env:
#   LOG_WAIT_SEC   Max seconds to wait for a new device log after run (default: 300)
#   ROUNDS            Number of timed rounds without --enable-profiling (default: 10)
#   ROUNDS_PROFILING  Number of timed rounds with --enable-profiling (default: 10)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$SCRIPT_DIR"
EXAMPLE_ARGS=(
    -k tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/kernels
    -g tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/golden.py
    -p a2a3 -d 11 --case Case1
)
# Must match -d in EXAMPLE_ARGS (wait_for_new_log polls $LOG_ROOT/device-${DEVICE_ID}/).
DEVICE_ID=11
# a2a3 cycle counter → microseconds (same as tools/benchmark_rounds.sh)
FREQ=50
ROUNDS="${ROUNDS:-10}"
ROUNDS_PROFILING="${ROUNDS_PROFILING:-10}"
LOG_WAIT_SEC="${LOG_WAIT_SEC:-300}"
# ---------------------------------------------------------------------------
# Device log directory (same as tools/benchmark_rounds.sh / device_log_resolver.py)
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
    if [[ ! -d "$LOG_ROOT" ]]; then
        LOG_ROOT="$HOME/ascend/log/debug"
    fi
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"
# ---------------------------------------------------------------------------
# parse_timing <log_file>  — copied from tools/benchmark_rounds.sh
# ---------------------------------------------------------------------------
parse_timing() {
    local log_file="$1"
    local timing
    timing=$(grep -E 'Thread [0-9]+: (sched_start|orch_start|orch_end|sched_end|orch_stage_end)' "$log_file" || true)
    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi
    echo "$timing" | awk -v freq="$FREQ" '
    function new_round() {
        flush_round()
        round++
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        delete sched_seen
        delete orch_seen
    }
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            results[round] = (max_end - min_start) / freq
            if (max_sched_end > 0 && min_sched_start > 0)
                sched_results[round] = (max_sched_end - min_sched_start) / freq
            if (max_orch_end > 0 && min_orch_start > 0)
                orch_results[round] = (max_orch_end - min_orch_start) / freq
            count++
        }
    }
    BEGIN {
        round = 0; count = 0
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        has_sched = 0; has_orch_end = 0
    }
    /sched_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in sched_seen) new_round()
        sched_seen[tid] = 1
        has_sched = 1
        match($0, /sched_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_sched_start == 0 || val < min_sched_start) min_sched_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /orch_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in orch_seen) new_round()
        orch_seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_orch_start == 0 || val < min_orch_start) min_orch_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /sched_end[^=]*=/ {
        match($0, /sched_end[^=]*=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_sched_end) max_sched_end = val
        if (val > max_end) max_end = val
    }
    /orch_end=/ {
        match($0, /orch_end=([0-9]+)/, m)
        val = m[1] + 0
        has_orch_end = 1
        if (val > max_orch_end) max_orch_end = val
        if (val > max_end) max_end = val
    }
    /orch_stage_end=/ {
        match($0, /orch_stage_end=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        flush_round()
        if (count == 0) { print "  (no rounds parsed)"; exit 1 }
        show_sched = has_sched
        show_orch = has_orch_end
        hdr = sprintf("  %-8s  %12s", "Round", "Elapsed (us)")
        sep = sprintf("  %-8s  %12s", "-----", "------------")
        if (show_sched) { hdr = hdr sprintf("  %12s", "Sched (us)"); sep = sep sprintf("  %12s", "----------") }
        if (show_orch)  { hdr = hdr sprintf("  %12s", "Orch (us)");  sep = sep sprintf("  %12s", "---------")  }
        print hdr; print sep
        sum_v = 0; min_v = results[0]; max_v = results[0]
        sum_s = 0; min_s = sched_results[0]; max_s = sched_results[0]
        sum_o = 0; min_o = orch_results[0]; max_o = orch_results[0]
        for (i = 0; i < count; i++) {
            line = sprintf("  %-8d  %12.1f", i, results[i])
            sum_v += results[i]
            if (results[i] < min_v) min_v = results[i]
            if (results[i] > max_v) max_v = results[i]
            if (show_sched) {
                line = line sprintf("  %12.1f", sched_results[i])
                sum_s += sched_results[i]
                if (sched_results[i] < min_s) min_s = sched_results[i]
                if (sched_results[i] > max_s) max_s = sched_results[i]
            }
            if (show_orch) {
                line = line sprintf("  %12.1f", orch_results[i])
                sum_o += orch_results[i]
                if (orch_results[i] < min_o) min_o = orch_results[i]
                if (orch_results[i] > max_o) max_o = orch_results[i]
            }
            print line
        }
        printf "\n  Avg: %.1f us", sum_v / count
        if (show_sched) printf "  |  Sched Avg: %.1f us", sum_s / count
        if (show_orch)  printf "  |  Orch Avg: %.1f us", sum_o / count
        printf "  (%d rounds)\n", count
    }'
}
# ---------------------------------------------------------------------------
# wait_for_new_log <pre_run_logs_file>  — adapted: uses LOG_WAIT_SEC
# ---------------------------------------------------------------------------
wait_for_new_log() {
    local pre_file="$1"
    local new_log=""
    local deadline=$((SECONDS + LOG_WAIT_SEC))
    while [[ $SECONDS -lt $deadline ]]; do
        if [[ -d "$DEVICE_LOG_DIR" ]]; then
            new_log=$(comm -13 "$pre_file" <(ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort) 2>/dev/null | tail -1 || true)
            if [[ -n "$new_log" ]]; then
                echo "$new_log"
                return 0
            fi
        fi
        sleep 0.5
    done
    if [[ -d "$DEVICE_LOG_DIR" ]]; then
        new_log=$(ls -t "$DEVICE_LOG_DIR"/*.log 2>/dev/null | head -1 || true)
        if [[ -n "$new_log" ]]; then
            echo "$new_log"
            return 0
        fi
    fi
    return 1
}
# Parse the single summary line from parse_timing ("  Avg: ... us  |  Sched Avg: ... |  Orch Avg: ...").
# Sets BENCH_AVG_ELAPSED_US; BENCH_SCHED_AVG_US / BENCH_ORCH_AVG_US if present (else empty).
parse_bench_summary_line() {
    local parse_out="$1"
    local line
    line=$(echo "$parse_out" | grep '^  Avg:' | head -1) || true
    if [[ -z "$line" ]]; then
        BENCH_AVG_ELAPSED_US=""
        BENCH_SCHED_AVG_US=""
        BENCH_ORCH_AVG_US=""
        return 1
    fi
    BENCH_AVG_ELAPSED_US=$(echo "$line" | sed -n 's/^  Avg: \([0-9.]*\) us.*/\1/p')
    BENCH_SCHED_AVG_US=$(echo "$line" | sed -n 's/.*Sched Avg: \([0-9.]*\) us.*/\1/p')
    BENCH_ORCH_AVG_US=$(echo "$line" | sed -n 's/.*Orch Avg: \([0-9.]*\) us.*/\1/p')
    [[ -n "$BENCH_AVG_ELAPSED_US" ]]
}
# Optional sched/orch fragments for final summary (empty parts omitted).
format_sched_orch_suffix() {
    local s="$1"
    local o="$2"
    local out=""
    if [[ -n "$s" ]]; then
        out+="  |  sched ${s} us"
    fi
    if [[ -n "$o" ]]; then
        out+="  |  orch ${o} us"
    fi
    echo "$out"
}
# Snapshot current log list, run command, wait for new log, print timing table; sets BENCH_* averages
run_timed_batch() {
    local title="$1"
    shift
    local pre_file
    pre_file=$(mktemp)
    trap 'rm -f -- "$pre_file"' RETURN
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$pre_file" || true
    echo ""
    echo "=== ${title} ==="
    python examples/scripts/run_example.py "$@"
    local new_log
    if ! new_log=$(wait_for_new_log "$pre_file"); then
        echo "ERROR: no device log under $DEVICE_LOG_DIR (waited ${LOG_WAIT_SEC}s)."
        BENCH_AVG_ELAPSED_US=""
        BENCH_SCHED_AVG_US=""
        BENCH_ORCH_AVG_US=""
        return 1
    fi
    echo "Device log: $new_log"
    local parse_out
    if ! parse_out=$(parse_timing "$new_log"); then
        echo "$parse_out"
        BENCH_AVG_ELAPSED_US=""
        BENCH_SCHED_AVG_US=""
        BENCH_ORCH_AVG_US=""
        return 1
    fi
    echo "$parse_out"
    if ! parse_bench_summary_line "$parse_out"; then
        echo "ERROR: could not parse average from timing output."
        BENCH_SCHED_AVG_US=""
        BENCH_ORCH_AVG_US=""
        return 1
    fi
    return 0
}
echo "=== Step 1: rm -rf build/cache ==="
rm -rf build/cache
echo "=== Step 2: pip install . ==="
CC=gcc CXX=g++ pip install .
echo "=== Step 3: warmup (1 run, no --enable-profiling) ==="
python examples/scripts/run_example.py "${EXAMPLE_ARGS[@]}"
AVG_NO_PROF=""
SCHED_NO_PROF=""
ORCH_NO_PROF=""
AVG_WITH_PROF=""
SCHED_WITH_PROF=""
ORCH_WITH_PROF=""
echo ""
echo "=== Step 4: timed batches (--skip-golden), parse device log ==="
echo "  without profiling: ${ROUNDS} rounds | with --enable-profiling: ${ROUNDS_PROFILING} rounds"
echo "Device log dir: $DEVICE_LOG_DIR"
if run_timed_batch "Without --enable-profiling (${ROUNDS} rounds)" \
    "${EXAMPLE_ARGS[@]}" -n "$ROUNDS" --skip-golden; then
    AVG_NO_PROF="$BENCH_AVG_ELAPSED_US"
    SCHED_NO_PROF="$BENCH_SCHED_AVG_US"
    ORCH_NO_PROF="$BENCH_ORCH_AVG_US"
fi
if run_timed_batch "With --enable-profiling (${ROUNDS_PROFILING} rounds)" \
    "${EXAMPLE_ARGS[@]}" -n "$ROUNDS_PROFILING" --skip-golden --enable-profiling; then
    AVG_WITH_PROF="$BENCH_AVG_ELAPSED_US"
    SCHED_WITH_PROF="$BENCH_SCHED_AVG_US"
    ORCH_WITH_PROF="$BENCH_ORCH_AVG_US"
fi
echo ""
echo "================================================================"
echo "  Summary (averages from device log, same as benchmark_rounds.sh)"
echo "================================================================"
if [[ -n "$AVG_NO_PROF" ]]; then
    suf=$(format_sched_orch_suffix "$SCHED_NO_PROF" "$ORCH_NO_PROF")
    echo "  Without --enable-profiling:  elapsed ${AVG_NO_PROF} us${suf}  (${ROUNDS} rounds)"
else
    echo "  Without --enable-profiling:  (failed to measure)"
fi
if [[ -n "$AVG_WITH_PROF" ]]; then
    suf=$(format_sched_orch_suffix "$SCHED_WITH_PROF" "$ORCH_WITH_PROF")
    echo "  With --enable-profiling:     elapsed ${AVG_WITH_PROF} us${suf}  (${ROUNDS_PROFILING} rounds)"
else
    echo "  With --enable-profiling:     (failed to measure)"
fi
echo "================================================================"
if [[ -z "$AVG_NO_PROF" || -z "$AVG_WITH_PROF" ]]; then
    exit 1
fi
