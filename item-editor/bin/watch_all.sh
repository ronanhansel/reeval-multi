#!/bin/bash
#
# Watch All - benchmark progress and logs (current run only)
#
# Usage:
#   ./watch_all.sh [--batch-mode] [--prefix PREFIX] [track_run_progress args...]
#   ./watch_all.sh logs
#
# Modes:
#   (default) Aggregate progress (same as previous agents view)
#   logs      Tail logs for the latest run only
#
# Batch mode:
#   - If finished tasks do not change for >7 minutes, launch next run immediately.
#   - If finished tasks == total tasks, launch next run after 5 minutes.
#   - Next prefix is computed by incrementing the current prefix (e.g., sun12_ -> sun13_).
#   - Next run command is previewed in the watcher.
#

show_help() {
    cat <<'EOF'
Usage:
  ./watch_all.sh [--batch-mode] [--prefix PREFIX] [track_run_progress args...]
  ./watch_all.sh logs

Modes:
  (default) Aggregate progress (same as previous agents view)
  logs      Tail logs for the latest run only

Options:
  --batch-mode     Auto-start the next colbench run if progress stalls or completes.
  --prefix PREFIX  Current run prefix (e.g., sun12_) used to compute the next prefix.
EOF
}

# Get script directory and repo root (script now lives in bin/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="agents"
BATCH_MODE="false"
PREFIX=""
TRACK_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        logs)
            MODE="logs"
            shift
            ;;
        --batch-mode)
            BATCH_MODE="true"
            shift
            ;;
        --prefix)
            PREFIX="${2:-}"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            TRACK_ARGS+=("$1")
            shift
            ;;
    esac
done

# Colors (log viewer only)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
BOLD_GREEN='\033[1;32m'
DIM='\033[2m'
NC='\033[0m'

RESULTS_DIR="$ROOT_DIR/results"
detect_run_root() {
    if [ -d "$ROOT_DIR/.hal_data" ] && [ -w "$ROOT_DIR/.hal_data" ]; then
        echo "$ROOT_DIR/.hal_data"
        return
    fi
    if [ -n "${DATA_PATH:-}" ] && [ -d "$DATA_PATH" ] && [ -w "$DATA_PATH" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$DATA_PATH/hal_runs/$namespace/$(basename "$ROOT_DIR")"
        return
    fi
    if [ -n "${HAL_DATA_ROOT:-}" ] && [ -d "$HAL_DATA_ROOT" ] && [ -w "$HAL_DATA_ROOT" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$HAL_DATA_ROOT/hal_runs/$namespace/$(basename "$ROOT_DIR")"
        return
    fi
    echo "$ROOT_DIR"
}

RUN_ROOT="$(detect_run_root)"

list_results_roots() {
    local roots=()
    local seen="|"
    add_root() {
        local dir="$1"
        [ -d "$dir" ] || return
        case "$seen" in
            *"|$dir|"*) return ;;
        esac
        roots+=("$dir")
        seen="${seen}${dir}|"
    }
    if [ -n "${HAL_RESULTS_DIR:-}" ]; then
        add_root "$HAL_RESULTS_DIR"
    fi
    add_root "$ROOT_DIR/.hal_data/results"
    add_root "$RUN_ROOT/results"
    add_root "$ROOT_DIR/.results"
    add_root "$ROOT_DIR/results"
    add_root "$ROOT_DIR/.result"
    add_root "$ROOT_DIR/result"
    printf "%s\n" "${roots[@]}"
}

# For backward compatibility and single-path use cases
RESULTS_DIR="$(list_results_roots | head -n 1)"

local_logs_root() {
    if [ -d "$ROOT_DIR/.hal_data/logs" ]; then
        echo "$ROOT_DIR/.hal_data/logs"
        return
    fi
    for candidate in "$ROOT_DIR/logs" "$ROOT_DIR/.logs" "$ROOT_DIR/log" "$ROOT_DIR/.log"; do
        if [ -L "$candidate" ] && [ ! -e "$candidate" ]; then
            continue
        fi
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
    echo "$ROOT_DIR/logs"
}

detect_logs_root() {
    if [ -n "${DATA_PATH:-}" ] && [ -d "$DATA_PATH" ] && [ -w "$DATA_PATH" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$DATA_PATH/hal_runs/$namespace/$(basename "$ROOT_DIR")/logs"
        return
    fi
    if [ -n "${HAL_DATA_ROOT:-}" ] && [ -d "$HAL_DATA_ROOT" ] && [ -w "$HAL_DATA_ROOT" ]; then
        local namespace="${HAL_DATA_NAMESPACE:-$USER}"
        echo "$HAL_DATA_ROOT/hal_runs/$namespace/$(basename "$ROOT_DIR")/logs"
        return
    fi
    local_logs_root
}

LOGS_DIR="$(detect_logs_root)"

list_logs_roots() {
    local roots=()
    local seen="|"
    add_root() {
        local dir="$1"
        [ -d "$dir" ] || return
        case "$seen" in
            *"|$dir|"*) return ;;
        esac
        roots+=("$dir")
        seen="${seen}${dir}|"
    }
    add_root "$ROOT_DIR/.hal_data/logs"
    add_root "$LOGS_DIR"
    add_root "$ROOT_DIR/logs"
    add_root "$ROOT_DIR/.logs"
    add_root "$ROOT_DIR/log"
    add_root "$ROOT_DIR/.log"
    printf "%s\n" "${roots[@]}"
}

get_latest_run_dir() {
    local roots=()
    local entries=()
    mapfile -t roots < <(list_logs_roots)
    if [ ${#roots[@]} -eq 0 ]; then
        return
    fi
    shopt -s nullglob
    for root in "${roots[@]}"; do
        for dir in "$root"/benchmark_run_*; do
            [ -d "$dir" ] || continue
            
            # Filter by prefix if provided
            if [ -n "$PREFIX" ]; then
                # Check config.json for prefix
                if [ -f "$dir/config.json" ]; then
                    local run_prefix
                    run_prefix=$(grep -o '"prefix": *"[^"]*"' "$dir/config.json" | cut -d'"' -f4)
                    if [[ "$run_prefix" != "$PREFIX" ]]; then
                        continue
                    fi
                fi
            fi
            
            local run_id
            run_id="$(basename "$dir" | sed 's/^benchmark_run_//')"
            entries+=("${run_id} ${dir}")
        done
    done
    shopt -u nullglob
    if [ ${#entries[@]} -eq 0 ]; then
        return
    fi
    printf "%s\n" "${entries[@]}" | sort -r | head -n 1 | cut -d' ' -f2-
}

get_latest_run_id() {
    local latest_run_dir
    latest_run_dir="$(get_latest_run_dir)"
    [ -n "$latest_run_dir" ] && basename "$latest_run_dir" | sed 's/^benchmark_run_//'
}

# Format and colorize function for log tailing
format_and_colorize() {
    awk -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v blue="$BLUE" \
        -v cyan="$CYAN" -v magenta="$MAGENTA" -v white="$WHITE" -v bold_green="$BOLD_GREEN" \
        -v nc="$NC" -v prefix_val="$PREFIX" '
    BEGIN {
        current_run_id = ""
    }
    /^==> .* <==/ {
        path = $2
        n = split(path, parts, "/")
        if (n >= 3) {
            current_benchmark = parts[n-2]
            current_run_id = parts[n-1]
        } else if (n >= 2) {
            current_benchmark = ""
            current_run_id = parts[n-1]
        } else {
            current_benchmark = ""
            current_run_id = path
        }
        next
    }
    /^$/ { next }
    {
        timestamp = strftime("%H:%M:%S")
        run = current_run_id
        
        if (current_run_id != "") {
            # Try to strip benchmark + prefix from the start
            # e.g. colbench_sun13_gpt-5.1-codex_... -> gpt-5.1-codex_...
            if (prefix_val != "") {
                p_idx = index(run, prefix_val)
                if (p_idx > 0) {
                    run = substr(run, p_idx + length(prefix_val))
                }
            } else {
                # Fallback: strip first two underscore-separated parts
                sub(/^[a-zA-Z0-9-]+_[a-zA-Z0-9-]+_/, "", run)
            }
        }
        
        prefix = sprintf("[%s %s] ", timestamp, run)

        line = $0
        gsub(/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+ - [a-zA-Z_.]+ - (DEBUG|INFO|WARNING|ERROR) - /, "", line)

        if (line ~ /Results:.*\{/ || line ~ /"accuracy"/ || line ~ /"score"/ || \
            line ~ /Evaluation completed/ || line ~ /successful_tasks/ || line ~ /failed_tasks/) {
            printf "%s%s%s%s\n", bold_green, prefix, line, nc
        }
        else if (tolower(line) ~ /error|exception|failed|traceback/) {
            printf "%s%s%s%s\n", red, prefix, line, nc
        } else if (tolower(line) ~ /401|403|429|500|502|503|504|timeout|unauthorized/) {
            printf "%s%s%s%s\n", magenta, prefix, line, nc
        } else if (tolower(line) ~ /success|completed|finished/) {
            printf "%s%s%s%s\n", green, prefix, line, nc
        } else if (tolower(line) ~ /warning|warn/) {
            printf "%s%s%s%s\n", yellow, prefix, line, nc
        } else if (tolower(line) ~ /starting|running|task/) {
            printf "%s%s%s%s\n", blue, prefix, line, nc
        } else if (line ~ /^\[hal\]/) {
            # Explicitly show [hal] verbose logs in cyan/white
            printf "%s%s%s\n", prefix, line, nc
        } else {
            printf "%s%s\n", prefix, line
        }
    }
    '
}

collect_logs() {
    local run_id="$1"
    local latest_run_dir="$2"
    local all_logs=""

    if [ -n "$PREFIX" ]; then
        # 1. Main logs for this prefix (search in all log roots)
        mapfile -t roots < <(list_logs_roots)
        for root in "${roots[@]}"; do
            # Find log dir that contains this prefix in config
            for dir in "$root"/benchmark_run_*; do
                [ -d "$dir" ] || continue
                if [ -f "$dir/config.json" ]; then
                    local p=$(grep -o '"prefix": *"[^"]*"' "$dir/config.json" | cut -d'"' -f4)
                    if [[ "$p" == "$PREFIX" ]]; then
                        for log in "$dir"/*.log; do
                            [ -f "$log" ] && all_logs="$all_logs $log"
                        done
                    fi
                fi
            done
        done

        # 2. Verbose logs for this prefix in all results dirs
        mapfile -t r_roots < <(list_results_roots)
        for r_root in "${r_roots[@]}"; do
            while IFS= read -r log_file; do
                 all_logs="$all_logs $log_file"
            done < <(find -L "$r_root" -maxdepth 5 -type f -name "*${PREFIX}*_verbose.log" 2>/dev/null)
        done

    else
        # Original logic (fallback to run_id if no prefix)
        if [ -n "$latest_run_dir" ]; then
            for log in "$latest_run_dir"/*.log; do
                [ -f "$log" ] && all_logs="$all_logs $log"
            done
        fi

        if [ -n "$run_id" ]; then
            # Find all verbose logs in all results dirs matching the run timestamp
            mapfile -t r_roots < <(list_results_roots)
            for r_root in "${r_roots[@]}"; do
                while IFS= read -r log_file; do
                     all_logs="$all_logs $log_file"
                done < <(find -L "$r_root" -maxdepth 5 -type f -name "*${run_id}*_verbose.log" 2>/dev/null)
            done
        fi
    fi

    echo "$all_logs"
}

watch_logs() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}           LOG VIEWER MODE (LATEST RUN ONLY)${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${BLUE}Results roots:${NC} $(list_results_roots | paste -sd ',' -)"
    echo -e "${BLUE}Logs roots:${NC} $(list_logs_roots | paste -sd ',' -)"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    local current_run_id=""
    while true; do
        local latest_run_dir
        local run_id
        latest_run_dir="$(get_latest_run_dir)"
        run_id="$(get_latest_run_id)"

        if [ -z "$run_id" ] && [ -z "$PREFIX" ]; then
            echo -e "${YELLOW}No benchmark run found yet. Waiting...${NC}"
            sleep 5
            continue
        fi

        if [ -n "$run_id" ] && [ "$run_id" != "$current_run_id" ]; then
            echo -e "${BLUE}Run ID:${NC} ${run_id}"
            current_run_id="$run_id"
        elif [ -z "$run_id" ] && [ -n "$PREFIX" ]; then
            # If we have a prefix but no run_id yet, use prefix for display
            if [ "$current_run_id" != "$PREFIX" ]; then
                echo -e "${BLUE}Prefix:${NC} ${PREFIX}"
                current_run_id="$PREFIX"
            fi
        fi

        LOG_FILES=$(collect_logs "$run_id" "$latest_run_dir")
        if [ -z "$LOG_FILES" ]; then
            echo -e "${YELLOW}No log files found for current search. Waiting...${NC}"
            sleep 5
            continue
        fi

        LOG_COUNT=$(echo $LOG_FILES | wc -w)
        echo -e "${CYAN}Watching $LOG_COUNT log files...${NC}"

        (
            tail -f $LOG_FILES | format_and_colorize
        ) &
        local tail_pid=$!

        while true; do
            sleep 2
            if ! kill -0 "$tail_pid" 2>/dev/null; then
                break
            fi
            
            # Check if run ID changed
            local new_run_id
            new_run_id="$(get_latest_run_id)"
            if [ -n "$new_run_id" ] && [ "$new_run_id" != "$current_run_id" ]; then
                kill "$tail_pid" 2>/dev/null || true
                wait "$tail_pid" 2>/dev/null || true
                current_run_id="$new_run_id"
                break
            fi
        done
    done
}

arg_present() {
    local needle="$1"
    shift
    for arg in "$@"; do
        if [ "$arg" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

case "$MODE" in
    logs)
        watch_logs
        ;;
    *)
        if ! arg_present "--watch" "${TRACK_ARGS[@]}"; then
            TRACK_ARGS+=(--watch)
        fi
        if ! arg_present "--interval" "${TRACK_ARGS[@]}"; then
            TRACK_ARGS+=(--interval 2)
        fi
        if [ "$BATCH_MODE" = "true" ]; then
            TRACK_ARGS+=(--batch-mode)
        fi
        if [ -n "$PREFIX" ]; then
            TRACK_ARGS+=(--prefix "$PREFIX")
        fi
        python3 "$ROOT_DIR/scripts/track_run_progress.py" "${TRACK_ARGS[@]}"
        ;;
esac
