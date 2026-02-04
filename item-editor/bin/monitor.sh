#!/bin/bash
# Benchmark Monitor - Shows progress for all configs of a benchmark
#
# Usage: ./monitor.sh <benchmark> [options]
#
# Options:
#   -i, --interactive   Interactive mode with log preview
#   --once              Run once and exit
#   --interval N        Refresh interval in seconds (default: 5)
#
# Interactive Controls:
#   1-9, 0    Select config 1-10 and show log
#   ↑/↓       Navigate configs
#   ←/→       Hide/show log preview
#   Enter     Toggle log preview
#   r         Force refresh
#   q         Quit
#
# Examples:
#   ./monitor.sh scicode              # Basic monitoring
#   ./monitor.sh scicode -i           # Interactive with log preview
#   ./monitor.sh scienceagentbench -i --interval 10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"

# Load .env if exists
if [ -f "${SCRIPT_DIR}/../.env" ]; then
    export $(grep -v '^#' "${SCRIPT_DIR}/../.env" | xargs)
fi

# Call Python monitor
python3 "${SCRIPT_DIR}/monitor.py" "$@"
