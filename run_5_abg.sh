#!/usr/bin/env bash
# Profiling level verification for aicpu_build_graph runtime (paged_attention_unroll).
#
# Runs paged_attention_unroll Case1 with profiling levels 0/1/2/3 interleaved,
# then prints a summary table comparing Elapsed/Sched/Orch latencies.
#
# Usage:
#   ./run_5_abg.sh [-d <device>] [-n <runs>] [-w <warmup>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_EXAMPLE="$SCRIPT_DIR/examples/scripts/run_example.py"
KERNELS_DIR="$SCRIPT_DIR/tests/st/a2a3/aicpu_build_graph/paged_attention_unroll/kernels"
GOLDEN_PY="$SCRIPT_DIR/tests/st/a2a3/aicpu_build_graph/paged_attention_unroll/golden.py"

# Defaults
DEVICE_ID=11
RUNS=5
WARMUP=2
PLATFORM=a2a3
CASE=Case1
FREQ=50  # a2a3 clock frequency in MHz

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--device)   DEVICE_ID="$2"; shift 2 ;;
        -n|--runs)     RUNS="$2";      shift 2 ;;
        -w|--warmup)   WARMUP="$2";    shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-d device] [-n runs] [-w warmup]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LEVELS=(0 1 2 3)

# ---------------------------------------------------------------------------
# Device log directory
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" && -d "$ASCEND_WORK_PATH/log/debug" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"

# ---------------------------------------------------------------------------
# wait_for_new_log <pre_run_logs_file>
# ---------------------------------------------------------------------------
wait_for_new_log() {
    local pre_file="$1"
    local deadline=$((SECONDS + 15))
    while [[ $SECONDS -lt $deadline ]]; do
        if [[ -d "$DEVICE_LOG_DIR" ]]; then
            local new_log
            new_log=$(comm -13 "$pre_file" <(ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort) 2>/dev/null | tail -1 || true)
            if [[ -n "$new_log" ]]; then
                echo "$new_log"
                return 0
            fi
        fi
        sleep 0.5
    done
    if [[ -d "$DEVICE_LOG_DIR" ]]; then
        ls -t "$DEVICE_LOG_DIR"/*.log 2>/dev/null | head -1 || true
    fi
}

# ---------------------------------------------------------------------------
# parse_single_round <log_file>
# ---------------------------------------------------------------------------
parse_single_round() {
    local log_file="$1"
    local timing
    timing=$(grep -E 'Thread [0-9]+: (sched_start|orch_start|orch_end|sched_end|orch_stage_end)' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "0 0 0"
        return
    fi

    echo "$timing" | awk -v freq="$FREQ" '
    BEGIN {
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
    }
    /sched_start=/ {
        match($0, /sched_start=([0-9]+)/, m); val = m[1] + 0
        if (min_sched_start == 0 || val < min_sched_start) min_sched_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /orch_start=/ {
        match($0, /orch_start=([0-9]+)/, m); val = m[1] + 0
        if (min_orch_start == 0 || val < min_orch_start) min_orch_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /sched_end[^=]*=/ {
        match($0, /sched_end[^=]*=([0-9]+)/, m); val = m[1] + 0
        if (val > max_sched_end) max_sched_end = val
        if (val > max_end) max_end = val
    }
    /orch_end=/ {
        match($0, /orch_end=([0-9]+)/, m); val = m[1] + 0
        if (val > max_orch_end) max_orch_end = val
        if (val > max_end) max_end = val
    }
    /orch_stage_end=/ {
        match($0, /orch_stage_end=([0-9]+)/, m); val = m[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        elapsed = (max_end > 0 && min_start > 0) ? (max_end - min_start) / freq : 0
        sched   = (max_sched_end > 0 && min_sched_start > 0) ? (max_sched_end - min_sched_start) / freq : 0
        orch    = (max_orch_end > 0 && min_orch_start > 0) ? (max_orch_end - min_orch_start) / freq : 0
        printf "%.1f %.1f %.1f\n", elapsed, sched, orch
    }'
}

# ---------------------------------------------------------------------------
# run_one <profiling_level>
# ---------------------------------------------------------------------------
run_one() {
    local level="$1"

    local pre_log_file
    pre_log_file=$(mktemp)
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$pre_log_file" 2>/dev/null || true

    python3 "$RUN_EXAMPLE" \
        -k "$KERNELS_DIR" -g "$GOLDEN_PY" \
        -p "$PLATFORM" -d "$DEVICE_ID" \
        --skip-golden --case "$CASE" \
        --enable-profiling "$level" > /dev/null 2>&1

    local new_log
    new_log=$(wait_for_new_log "$pre_log_file")
    rm -f "$pre_log_file"

    if [[ -n "$new_log" ]]; then
        parse_single_round "$new_log"
    else
        echo "0 0 0"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Profiling Level Verification (aicpu_build_graph)"
echo "  Test:    paged_attention_unroll / $CASE"
echo "  Device:  $DEVICE_ID"
echo "  Warmup:  $WARMUP rounds"
echo "  Runs:    $RUNS per level (interleaved)"
echo "  Levels:  ${LEVELS[*]}"
echo "================================================================"

echo ""
echo "--- Warmup ($WARMUP rounds) ---"
for ((w = 1; w <= WARMUP; w++)); do
    printf "  Warmup %d/%d ...\r" "$w" "$WARMUP"
    run_one 0 > /dev/null
done
echo "  Warmup done.              "

declare -A RESULTS

echo ""
echo "--- Benchmark (interleaved, $RUNS runs x ${#LEVELS[@]} levels) ---"
for ((run = 1; run <= RUNS; run++)); do
    for level in "${LEVELS[@]}"; do
        printf "  Run %d/%d, level %d ...\r" "$run" "$RUNS" "$level"
        result=$(run_one "$level")
        RESULTS["$level,$run"]="$result"
    done
done
echo "  Benchmark done.                      "

echo ""
echo "================================================================"
echo "  Final Results -- interleaved 0/1/2/3, $RUNS runs each"
echo "================================================================"

for level in "${LEVELS[@]}"; do
    echo ""
    echo "--enable-profiling $level (interleaved, $RUNS runs):"
    printf "  %-5s  %12s  %10s  %10s\n" "Run" "Elapsed(us)" "Sched(us)" "Orch(us)"

    sum_e=0 sum_s=0 sum_o=0
    for ((run = 1; run <= RUNS; run++)); do
        read -r e s o <<< "${RESULTS[$level,$run]}"
        printf "  %-5d  %12s  %10s  %10s\n" "$run" "$e" "$s" "$o"
        sum_e=$(awk "BEGIN{print $sum_e + $e}")
        sum_s=$(awk "BEGIN{print $sum_s + $s}")
        sum_o=$(awk "BEGIN{print $sum_o + $o}")
    done

    avg_e=$(awk "BEGIN{printf \"%.1f\", $sum_e / $RUNS}")
    avg_s=$(awk "BEGIN{printf \"%.1f\", $sum_s / $RUNS}")
    avg_o=$(awk "BEGIN{printf \"%.1f\", $sum_o / $RUNS}")
    echo "  ------"
    printf "  Avg   %12s  %10s  %10s\n" "$avg_e" "$avg_s" "$avg_o"
done

echo ""
echo "================================================================"
echo "  Overhead Summary (avg vs level 0)"
echo "================================================================"
printf "  %-8s  %12s  %12s  %12s\n" "Level" "Elapsed(us)" "Sched(us)" "Orch(us)"
printf "  %-8s  %12s  %12s  %12s\n" "-----" "------------" "------------" "------------"

declare -A AVGS
for level in "${LEVELS[@]}"; do
    se=0 ss=0 so=0
    for ((run = 1; run <= RUNS; run++)); do
        read -r e s o <<< "${RESULTS[$level,$run]}"
        se=$(awk "BEGIN{print $se + $e}")
        ss=$(awk "BEGIN{print $ss + $s}")
        so=$(awk "BEGIN{print $so + $o}")
    done
    AVGS["$level,e"]=$(awk "BEGIN{printf \"%.1f\", $se / $RUNS}")
    AVGS["$level,s"]=$(awk "BEGIN{printf \"%.1f\", $ss / $RUNS}")
    AVGS["$level,o"]=$(awk "BEGIN{printf \"%.1f\", $so / $RUNS}")
done

base_e="${AVGS[0,e]}"
base_s="${AVGS[0,s]}"
base_o="${AVGS[0,o]}"

for level in "${LEVELS[@]}"; do
    ae="${AVGS[$level,e]}"
    as="${AVGS[$level,s]}"
    ao="${AVGS[$level,o]}"
    if [[ "$level" -eq 0 ]]; then
        printf "  %-8s  %12s  %12s  %12s  (baseline)\n" "$level" "$ae" "$as" "$ao"
    else
        de=$(awk "BEGIN{printf \"%+.1f\", $ae - $base_e}")
        ds=$(awk "BEGIN{printf \"%+.1f\", $as - $base_s}")
        do_=$(awk "BEGIN{printf \"%+.1f\", $ao - $base_o}")
        pe=$(awk "BEGIN{printf \"%+.1f%%\", ($ae - $base_e) / $base_e * 100}")
        ps=$(awk "BEGIN{printf \"%+.1f%%\", ($as - $base_s) / $base_s * 100}")
        po=$(awk "BEGIN{printf \"%+.1f%%\", ($ao - $base_o) / $base_o * 100}")
        printf "  %-8s  %8s %5s  %8s %5s  %8s %5s\n" "$level" "$ae" "$pe" "$as" "$ps" "$ao" "$po"
    fi
done

echo ""
echo "Done."
