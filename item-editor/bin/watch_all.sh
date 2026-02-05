#!/bin/bash
#
# Watch Logs - Simplified log tailing for a specific run prefix.
#
# Usage:
#   ./watch_all.sh --prefix PREFIX
#

show_help() {
    cat <<'EOF'
Usage:
  ./watch_all.sh --prefix PREFIX

Simplified log viewer that tails all logs matching a specific run prefix.
EOF
}

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX=""

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)
            PREFIX="${2:-}"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$PREFIX" ]; then
    echo "Error: --prefix is required."
    show_help
    exit 1
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD_GREEN='\033[1;32m'

# Discovery functions
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
    echo "$ROOT_DIR"
}

RUN_ROOT="$(detect_run_root)"

list_results_roots() {
    local roots=()
    [ -d "$ROOT_DIR/.hal_data/results" ] && roots+=("$ROOT_DIR/.hal_data/results")
    [ -d "$RUN_ROOT/results" ] && roots+=("$RUN_ROOT/results")
    [ -d "$ROOT_DIR/results" ] && roots+=("$ROOT_DIR/results")
    printf "%s\n" "${roots[@]}" | sort -u
}

list_logs_roots() {
    local roots=()
    [ -d "$ROOT_DIR/.hal_data/logs" ] && roots+=("$ROOT_DIR/.hal_data/logs")
    [ -d "$ROOT_DIR/logs" ] && roots+=("$ROOT_DIR/logs")
    printf "%s\n" "${roots[@]}" | sort -u
}

format_and_colorize() {
    awk -v red="$RED" -v green="$GREEN" -v yellow="$YELLOW" -v blue="$BLUE" \
        -v cyan="$CYAN" -v magenta="$MAGENTA" -v bold_green="$BOLD_GREEN" \
        -v nc="$NC" -v prefix_val="$PREFIX" '
    /^==> .* <==/ {
        path = $2
        n = split(path, parts, "/")
        current_run_id = (n >= 1) ? parts[n-1] : path
        next
    }
    /^$/ { next }
    {
        timestamp = strftime("%H:%M:%S")
        run = current_run_id
        
        if (prefix_val != "") {
            p_idx = index(run, prefix_val)
            if (p_idx > 0) {
                run = substr(run, p_idx + length(prefix_val))
            }
        }
        
        display_prefix = sprintf("[%s %s] ", timestamp, run)
        line = $0
        gsub(/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+ - [a-zA-Z_.]+ - (DEBUG|INFO|WARNING|ERROR) - /, "", line)

        if (line ~ /Results:.*\{/ || line ~ /"accuracy"/ || line ~ /"score"/ || \
            line ~ /Evaluation completed/ || line ~ /successful_tasks/ || line ~ /failed_tasks/) {
            printf "%s%s%s%s\n", bold_green, display_prefix, line, nc
        }
        else if (tolower(line) ~ /error|exception|failed|traceback/) {
            printf "%s%s%s%s\n", red, display_prefix, line, nc
        } else if (tolower(line) ~ /401|403|429|500|502|503|504|timeout|unauthorized/) {
            printf "%s%s%s%s\n", magenta, display_prefix, line, nc
        } else if (tolower(line) ~ /success|completed|finished/) {
            printf "%s%s%s%s\n", green, display_prefix, line, nc
        } else if (tolower(line) ~ /warning|warn/) {
            printf "%s%s%s%s\n", yellow, display_prefix, line, nc
        } else if (tolower(line) ~ /starting|running|task/) {
            printf "%s%s%s%s\n", blue, display_prefix, line, nc
        } else {
            printf "%s%s%s\n", display_prefix, line, nc
        }
    }
    '
}

collect_logs() {
    local all_logs=""

    # 1. Benchmark run logs
    mapfile -t l_roots < <(list_logs_roots)
    for root in "${l_roots[@]}"; do
        for dir in "$root"/benchmark_run_*;
 do
            [ -d "$dir" ] || continue
            if [ -f "$dir/config.json" ]; then
                local p=$(grep -o '"prefix": *"[^"]*"' "$dir/config.json" | cut -d'"' -f4)
                if [[ "$p" == "$PREFIX" ]]; then
                    for log in "$dir"/*.log;
 do
                        [ -f "$log" ] && all_logs="$all_logs $log"
                    done
                fi
            fi
        done
    done

    # 2. Results verbose logs
    mapfile -t r_roots < <(list_results_roots)
    for r_root in "${r_roots[@]}"; do
        while IFS= read -r log_file; do
             all_logs="$all_logs $log_file"
        done < <(find -L "$r_root" -maxdepth 5 -type f -name "*${PREFIX}*_verbose.log" 2>/dev/null)
    done

    echo "$all_logs"
}

watch_logs() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}           LOG VIEWER MODE (Prefix: $PREFIX)${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    while true; do
        LOG_FILES=$(collect_logs)
        if [ -z "$LOG_FILES" ]; then
            echo -e "${YELLOW}No log files found for prefix '$PREFIX'. Waiting...${NC}"
            sleep 5
            continue
        fi

        LOG_COUNT=$(echo $LOG_FILES | wc -w)
        echo -e "${CYAN}Watching $LOG_COUNT log files...${NC}"

        tail -f $LOG_FILES | format_and_colorize
        
        # If tail exits (e.g. files changed), wait and retry
        sleep 2
    done
}

watch_logs