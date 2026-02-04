#!/bin/bash
#
# Comprehensive Benchmark Runner
# Runs all benchmarks concurrently with centralized logging
#
# Usage: ./run_all_benchmarks.sh [options] [prefix] [parallel]
#
# Options:
#   --prefix PFX  Prefix for run IDs and output files (default: moon1_)
#   --benchmarks  Comma-separated list of benchmarks to run (e.g., colbench,scicode)
#   --parallel    Total parallel tasks to run (caps at 800, ignores --parallel-models)
#   --parallel-models N  Number of model configs to run concurrently
#   --parallel-tasks N   Number of tasks to run concurrently per model
#   --trace-mode MODE    Set HAL_TRACE_MODE (e.g., local)
#   --sample-tasks N     Randomly sample N tasks from each benchmark dataset
#   --sample-seed N      Seed for --sample-tasks to make selection reproducible
#   --fix-only     Run only tasks with fixes (skip --all-tasks)
#   --repeat N           Number of additional iterations to run (auto-increments prefix)
#   --until N            Keep running until the numeric part of prefix exceeds N
#   --next-parallel-tasks N  Set --parallel-tasks for subsequent iterations (repeats)
#
# Examples:
#   ./run_all_benchmarks.sh sun16_ 20           # Fresh run or resume of sun16_
#   ./run_all_benchmarks.sh sun16_ --repeat 3   # Run sun16_, sun17_, sun18_, sun19_
#

set -o pipefail

# Get script directory (works on different machines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${HAL_REPO_ROOT:-}" ]; then
    REPO_ROOT="$HAL_REPO_ROOT"
elif [ -d "$SCRIPT_DIR/../hal-harness" ] || [ -d "$SCRIPT_DIR/../scripts" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_ROOT="$SCRIPT_DIR"
fi
cd "$REPO_ROOT"

# Log directory detection (respects DATA_PATH/HAL_DATA_ROOT)
local_logs_root() {
    local candidate="$REPO_ROOT/logs"
    if [ -L "$candidate" ] && [ ! -e "$candidate" ]; then
        echo "$REPO_ROOT/.logs"
        return
    fi
    echo "$candidate"
}

detect_logs_root() {
    if [ -n "${DATA_PATH:-}" ] && [ -d "$DATA_PATH" ] && [ -w "$DATA_PATH" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$DATA_PATH/hal_runs/$namespace/$(basename "$REPO_ROOT")/logs"
        return
    fi
    if [ -n "${HAL_DATA_ROOT:-}" ] && [ -d "$HAL_DATA_ROOT" ] && [ -w "$HAL_DATA_ROOT" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$HAL_DATA_ROOT/hal_runs/$namespace/$(basename "$REPO_ROOT")/logs"
        return
    fi
    local_logs_root
}

LOGS_BASE="$(detect_logs_root)"
if ! mkdir -p "$LOGS_BASE" 2>/dev/null; then
    LOGS_BASE="$(local_logs_root)"
    mkdir -p "$LOGS_BASE" 2>/dev/null || true
fi

# Parse arguments
PREFIX=""
PREFIX_FROM_ARG=false
PARALLEL=""
PARALLEL_TOTAL=""
PARALLEL_FLAG=false
PARALLEL_MODELS=""
PARALLEL_TASKS=""
TRACE_MODE=""
SAMPLE_TASKS=""
SAMPLE_SEED=""
REPEAT_COUNT=0
UNTIL_NUM=""
NEXT_PARALLEL_TASKS=""
REQUESTED_BENCHMARKS=()
SKIP_ROOT_CHECK=false
FIX_ONLY=false
NO_FIX=false
FORCE_RESET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-reset)
            FORCE_RESET=true
            shift
            ;;
        --continue|--resume)
            # Legacy flags ignored, now behavior is automatic based on prefix
            shift
            ;;
        --prefix)
            shift
            PREFIX="${1:-}"
            PREFIX_FROM_ARG=true
            shift
            ;;
        --benchmarks|--tags)
            shift
            IFS=',' read -r -a REQUESTED_BENCHMARKS <<< "${1:-}"
            shift
            ;;
        --parallel)
            shift
            PARALLEL_TOTAL="${1:-}"
            PARALLEL_FLAG=true
            shift
            ;;
        --parallel-models)
            shift
            PARALLEL_MODELS="${1:-}"
            shift
            ;;
        --parallel-tasks)
            shift
            PARALLEL_TASKS="${1:-}"
            shift
            ;;
        --next-parallel-tasks)
            shift
            NEXT_PARALLEL_TASKS="${1:-}"
            shift
            ;;
        --trace-mode)
            shift
            TRACE_MODE="${1:-}"
            shift
            ;;
        --sample-tasks)
            shift
            SAMPLE_TASKS="${1:-}"
            shift
            ;;
        --sample-seed)
            shift
            SAMPLE_SEED="${1:-}"
            shift
            ;;
        --fix-only)
            FIX_ONLY=true
            shift
            ;;
        --no-fix)
            NO_FIX=true
            shift
            ;;
        --timeout)
            shift
            TIMEOUT="${1:-}"
            shift
            ;;
        --repeat)
            shift
            REPEAT_COUNT="${1:-}"
            shift
            ;;
        --skip-root-check)
            SKIP_ROOT_CHECK=true
            shift
            ;;
        --until)
            shift
            UNTIL_NUM="${1:-}"
            shift
            ;;
        *)
            if [ -z "$PREFIX" ]; then
                PREFIX="$1"
                PREFIX_FROM_ARG=true
            elif [ -z "$PARALLEL" ]; then
                PARALLEL="$1"
            fi
            shift
            ;;
    esac
done

# Defaults
PREFIX="${PREFIX:-moon1_}"
PARALLEL="${PARALLEL:-10}"

if [ "$FORCE_RESET" = true ]; then
    echo -e "${RED}[reset] Force reset enabled. Deleting previous logs and results for prefix: $PREFIX${NC}"
    # Use a safe pattern matching to avoid accidental deletions
    # Targets benchmark_run_<prefix>* in logs
    # Targets <benchmark>_<prefix>* in results
    rm -rf "$LOGS_BASE"/benchmark_run_"${PREFIX}"* 2>/dev/null || true
    rm -rf "$REPO_ROOT"/results/*/*"${PREFIX}"* 2>/dev/null || true
    echo -e "${GREEN}[reset] Cleanup complete.${NC}"
fi

PARALLEL_MODELS="${PARALLEL_MODELS:-$PARALLEL}"
PARALLEL_TASKS="${PARALLEL_TASKS:-$PARALLEL}"
PARALLEL_CAP=800

is_positive_int() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

if $PARALLEL_FLAG; then
    if ! is_positive_int "$PARALLEL_TOTAL"; then
        echo "Invalid --parallel value: '$PARALLEL_TOTAL' (expected positive integer)"
        exit 1
    fi
    if [ "$PARALLEL_TOTAL" -gt "$PARALLEL_CAP" ]; then
        echo "Capping --parallel from $PARALLEL_TOTAL to $PARALLEL_CAP for stability."
        PARALLEL_TOTAL="$PARALLEL_CAP"
    fi
    if ! is_positive_int "$PARALLEL_MODELS"; then
        echo "Invalid --parallel-models value: '$PARALLEL_MODELS' (expected positive integer)"
        exit 1
    fi
    if [ "$PARALLEL_MODELS" -gt "$PARALLEL_TOTAL" ]; then
        PARALLEL_MODELS="$PARALLEL_TOTAL"
    fi
    PARALLEL_TASKS=$(( (PARALLEL_TOTAL + PARALLEL_MODELS - 1) / PARALLEL_MODELS ))
    if [ "$PARALLEL_TASKS" -lt 1 ]; then
        PARALLEL_TASKS=1
    fi
    PARALLEL_MODELS=$(( PARALLEL_TOTAL / PARALLEL_TASKS ))
    if [ "$PARALLEL_MODELS" -lt 1 ]; then
        PARALLEL_MODELS=1
    fi
else
    if ! is_positive_int "$PARALLEL_MODELS"; then
        echo "Invalid --parallel-models value: '$PARALLEL_MODELS' (expected positive integer)"
        exit 1
    fi
    if ! is_positive_int "$PARALLEL_TASKS"; then
        echo "Invalid --parallel-tasks value: '$PARALLEL_TASKS' (expected positive integer)"
        exit 1
    fi
fi

# Use host networking to avoid docker0 bridge limits (allows 1000+ containers)
export HAL_DOCKER_NETWORK_MODE=host

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Benchmarks to run
ALL_BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")

benchmark_in_list() {
    local target="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$target" ]; then
            return 0
        fi
    done
    return 1
}

if [ ${#REQUESTED_BENCHMARKS[@]} -gt 0 ]; then
    for b in "${REQUESTED_BENCHMARKS[@]}"; do
        if ! benchmark_in_list "$b" "${ALL_BENCHMARKS[@]}"; then
            echo "Unknown benchmark: $b"
            echo "Valid benchmarks: ${ALL_BENCHMARKS[*]}"
            exit 1
        fi
    done
fi

# =============================================================================
# Helper Functions
# =============================================================================

# Function to increment prefix (sun16_ -> sun17_)
increment_prefix() {
    local pfx="$1"
    if [[ "$pfx" =~ ([^0-9]*)([0-9]+)([^0-9]*) ]]; then
        local base="${BASH_REMATCH[1]}"
        local num="${BASH_REMATCH[2]}"
        local suffix="${BASH_REMATCH[3]}"
        local next_num=$((10#$num + 1))
        echo "${base}${next_num}${suffix}"
    else
        echo "${pfx}1"
    fi
}

get_prefix_num() {
    local pfx="$1"
    if [[ "$pfx" =~ ([^0-9]*)([0-9]+)([^0-9]*) ]]; then
        echo $((10#${BASH_REMATCH[2]}))
    else
        echo "-1"
    fi
}

benchmarks_key() {
    local joined=""
    local item
    for item in "$@"; do
        if [ -z "$joined" ]; then
            joined="$item"
        else
            joined="${joined}+${item}"
        fi
    done
    echo "$joined"
}

find_log_dir_for_prefix_and_benchmarks() {
    local log_base="$1"
    local target_prefix="$2"
    local target_bench_key="$3"
    local dir
    # Look for benchmark_run_<prefix>__<benchmarks_key>_*
    if [ -d "$log_base/benchmark_run_${target_prefix}__${target_bench_key}" ]; then
        echo "$log_base/benchmark_run_${target_prefix}__${target_bench_key}"
        return 0
    fi
    for dir in $(ls -td "$log_base"/benchmark_run_* 2>/dev/null); do
        local cfg="$dir/config.json"
        [ -f "$cfg" ] || continue
        local info
        info=$(python3 - "$cfg" <<'PY'
import json, sys
from pathlib import Path
cfg = Path(sys.argv[1])
try:
    data = json.loads(cfg.read_text())
except Exception:
    print("", end="")
    sys.exit(0)
prefix = data.get("prefix", "")
benchmarks = data.get("benchmarks", [])
bench_key = "+".join(benchmarks) if isinstance(benchmarks, list) else ""
print(f"{prefix}|{bench_key}", end="")
PY
)
        [ -n "$info" ] || continue
        local saved_prefix="${info%%|*}"
        local saved_bench_key="${info#*|}"
        if [ "$saved_prefix" = "$target_prefix" ] && [ "$saved_bench_key" = "$target_bench_key" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

# =============================================================================
# PRE-BUILD FUNCTION
# =============================================================================

run_comprehensive_prebuild() {
    local benchmarks="$@"
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}[$(date +%H:%M:%S)] PHASE 1: PRE-BUILDING DOCKER IMAGES${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${BLUE}Benchmarks:${NC} ${benchmarks:-ALL}"
    echo ""

    if [ -f "$SCRIPT_DIR/prebuild_all_images.sh" ]; then
        if bash "$SCRIPT_DIR/prebuild_all_images.sh" $benchmarks; then
            echo -e "${GREEN}[prebuild] All Docker images ready!${NC}"
            return 0
        else
            echo -e "${RED}[prebuild] CRITICAL: Prebuild failed!${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}[prebuild] prebuild_all_images.sh not found, skipping...${NC}"
        return 0
    fi
}

# =============================================================================
# MAIN LOOP
# =============================================================================

ACTIVE_BENCHMARKS=("scicode" "scienceagentbench" "corebench" "colbench")
if [ ${#REQUESTED_BENCHMARKS[@]} -gt 0 ]; then
    ACTIVE_BENCHMARKS=("${REQUESTED_BENCHMARKS[@]}")
fi

if ! run_comprehensive_prebuild "${ACTIVE_BENCHMARKS[@]}"; then
    exit 1
fi

# Ensure CoreBench data is downloaded if CoreBench is active
if benchmark_in_list "corebench" "${ACTIVE_BENCHMARKS[@]}"; then
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}[$(date +%H:%M:%S)] PHASE 1.5: ENSURING COREBENCH DATA${NC}"
    echo -e "${CYAN}================================================================${NC}"
    if python3 "$REPO_ROOT/scripts/ensure_corebench_data.py"; then
        echo -e "${GREEN}[data] CoreBench data ready.${NC}"
    else
        echo -e "${RED}[data] Failed to ensure CoreBench data!${NC}"
        exit 1
    fi
fi

# Print task count summary
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}[$(date +%H:%M:%S)] PHASE 2: TASK COUNT SUMMARY${NC}"
echo -e "${CYAN}================================================================${NC}"
benchmarks_list=$(IFS=,; echo "${ACTIVE_BENCHMARKS[*]}")
if $NO_FIX; then
    python3 "$REPO_ROOT/scripts/get_task_counts.py" --benchmarks "$benchmarks_list" --all-tasks
elif $FIX_ONLY; then
    python3 "$REPO_ROOT/scripts/get_task_counts.py" --benchmarks "$benchmarks_list"
else
    python3 "$REPO_ROOT/scripts/get_task_counts.py" --benchmarks "$benchmarks_list" --all-tasks
fi

while true; do
    echo -e "${CYAN}       COMPREHENSIVE BENCHMARK RUNNER${NC}"
    echo -e "${CYAN}============================================================${NC}"
    
    # Prefix + benchmarks deterministic Log Directory
    BENCHMARKS_KEY="$(benchmarks_key "${ACTIVE_BENCHMARKS[@]}")"
    LOG_DIR=$(find_log_dir_for_prefix_and_benchmarks "$LOGS_BASE" "$PREFIX" "$BENCHMARKS_KEY")
    IS_CONTINUE=false

    if [ -n "$LOG_DIR" ]; then
        echo -e "${YELLOW}Detected existing run for prefix '$PREFIX' and benchmarks '$BENCHMARKS_KEY'. Resuming...${NC}"
        IS_CONTINUE=true
    else
        # New directory named by prefix and benchmarks
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_DIR="$LOGS_BASE/benchmark_run_${PREFIX}__${BENCHMARKS_KEY}_${TIMESTAMP}"
        mkdir -p "$LOG_DIR"
    fi

    # Determine benchmarks to run/resume
    BENCHMARKS_TO_RUN=()

    for benchmark in "${ACTIVE_BENCHMARKS[@]}"; do
        if $IS_CONTINUE;
 then
            exit_code_file="$LOG_DIR/${benchmark}.exit_code"
            if [ -f "$exit_code_file" ] && [ "$(cat "$exit_code_file")" -eq 0 ]; then
                echo -e "  ${GREEN}[OK]${NC} $benchmark - already completed"
                continue
            fi
        fi
        BENCHMARKS_TO_RUN+=("$benchmark")
    done

    if [ ${#BENCHMARKS_TO_RUN[@]} -eq 0 ]; then
        echo -e "${GREEN}All requested benchmarks for prefix '$PREFIX' are already complete.${NC}"
    else
        echo -e "${BLUE}Prefix:${NC} $PREFIX"
        echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
        echo -e "${BLUE}Parallel Models:${NC} $PARALLEL_MODELS"
        echo -e "${BLUE}Parallel Tasks:${NC} $PARALLEL_TASKS"
        if $NO_FIX; then
            echo -e "${BLUE}Mode:${NC} all tasks (NO FIXES)"
        elif $FIX_ONLY; then
            echo -e "${BLUE}Mode:${NC} fixes only"
        else
            echo -e "${BLUE}Mode:${NC} all tasks"
        fi
        echo -e "${BLUE}Benchmarks:${NC} ${BENCHMARKS_TO_RUN[*]}"
        echo -e "${CYAN}============================================================${NC}"
        echo ""

        # Save config
        benchmarks_json=$(printf '"%s",' "${ACTIVE_BENCHMARKS[@]}")
        benchmarks_json="[${benchmarks_json%,}]"
        cat > "$LOG_DIR/config.json" << EOF
{
    "prefix": "$PREFIX",
    "parallel_models": $PARALLEL_MODELS,
    "parallel_tasks": $PARALLEL_TASKS,
    "benchmarks": $benchmarks_json
}
EOF

        # Run logic
        filter_log() {
            local benchmark=$1
            local color=$2
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue
                if echo "$line" | grep -qiE "error|exception|failed|traceback|401|unauthorized"; then
                    echo -e "${RED}[$benchmark] $line${NC}"
                elif echo "$line" | grep -qiE "COMPLETED|FINISHED|All.*done"; then
                    echo -e "${GREEN}[$benchmark] $line${NC}"
                elif echo "$line" | grep -q "STEP:"; then
                     echo -e "${color}[$benchmark] $line${NC}"
                fi
            done
        }

        run_benchmark() {
            local benchmark=$1
            local color=$2
            local log_file="$LOG_DIR/${benchmark}.log"
            local pid_file="$LOG_DIR/${benchmark}.pid"
            local extra_args=(--resume) # Always resume within a prefix

            if [ -n "$TRACE_MODE" ]; then extra_args+=(--trace-mode "$TRACE_MODE"); fi
            if [ -n "$SAMPLE_TASKS" ]; then extra_args+=(--sample-tasks "$SAMPLE_TASKS"); fi
            if [ -n "$SAMPLE_SEED" ]; then extra_args+=(--sample-seed "$SAMPLE_SEED"); fi
            if [ -n "$TIMEOUT" ]; then extra_args+=(--timeout "$TIMEOUT"); fi

            (
                cd "$REPO_ROOT"
                touch "$log_file"
                tail -f -n 0 "$log_file" | filter_log "$benchmark" "$color" &
                local tail_pid=$!

                # Important: Use the same prefix naming convention that was used in previous runs
                # Previous runs used "${benchmark}_${PREFIX}" as the prefix passed to run_benchmark_fixes.py
                # e.g. colbench_sun16_
                local sub_prefix="${benchmark}_${PREFIX}"

                local cmd=(
                    "$SCRIPT_DIR/run_benchmark_with_data.sh"
                    python -u "$REPO_ROOT/scripts/run_benchmark_fixes.py"
                    --benchmark "$benchmark"
                    --all-configs
                    --prefix "$sub_prefix"
                    --docker
                    --parallel-models "$PARALLEL_MODELS"
                    --parallel-tasks "$PARALLEL_TASKS"
                    "${extra_args[@]}"
                )
                if $NO_FIX; then
                    cmd+=(--no-fix)
                elif ! $FIX_ONLY; then
                    cmd+=(--all-tasks)
                fi
                if $SKIP_ROOT_CHECK || [ -n "${HAL_SKIP_ROOT_CHECK:-}" ]; then
                    env HAL_SKIP_ROOT_CHECK=1 "${cmd[@]}" >> "$log_file" 2>&1 &
                else
                    "${cmd[@]}" >> "$log_file" 2>&1 &
                fi
                local cmd_pid=$!

                wait $cmd_pid
                exit_code=$?
                kill $tail_pid 2>/dev/null
                echo "$exit_code" > "$LOG_DIR/${benchmark}.exit_code"
            ) &
            echo $! > "$pid_file"
        }

        declare -A BENCHMARK_COLORS
        BENCHMARK_COLORS["scicode"]="$BLUE"
        BENCHMARK_COLORS["scienceagentbench"]="$GREEN"
        BENCHMARK_COLORS["corebench"]="$YELLOW"
        BENCHMARK_COLORS["colbench"]="$CYAN"

        for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
            # Set environment variables for specific benchmarks
            if [ "$benchmark" == "scicode" ]; then
                export SCICODE_EVAL_IMAGE="scicode-eval:latest"
                export SCICODE_EVAL_SKIP_INSTALL="1"
            else
                unset SCICODE_EVAL_IMAGE
                unset SCICODE_EVAL_SKIP_INSTALL
            fi

            run_benchmark "$benchmark" "${BENCHMARK_COLORS[$benchmark]:-$WHITE}"
            sleep 2
        done

        trap "exit 1" SIGTERM

        # Wait for this batch
        for benchmark in "${BENCHMARKS_TO_RUN[@]}"; do
            pid_file="$LOG_DIR/${benchmark}.pid"
            [ -f "$pid_file" ] && wait $(cat "$pid_file") 2>/dev/null
        done
    fi

    # Loop logic
    SHOULD_LOOP=false
    if [ "$REPEAT_COUNT" -gt 0 ]; then
        REPEAT_COUNT=$((REPEAT_COUNT - 1))
        SHOULD_LOOP=true
    elif [ -n "$UNTIL_NUM" ]; then
        current_num=$(get_prefix_num "$PREFIX")
        # Ensure current_num is a number before comparing
        if [[ "$current_num" =~ ^-?[0-9]+$ ]]; then
             if [ "$current_num" -ge 0 ] && [ "$current_num" -lt "$UNTIL_NUM" ]; then
                 SHOULD_LOOP=true
             fi
        fi
    fi
    
    if $SHOULD_LOOP; then
        PREFIX=$(increment_prefix "$PREFIX")
        [ -n "$NEXT_PARALLEL_TASKS" ] && PARALLEL_TASKS="$NEXT_PARALLEL_TASKS"
        LOG_DIR=""
    else
        break
    fi
done

echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}All iterations complete.${NC}"
exit 0
