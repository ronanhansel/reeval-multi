#!/bin/bash
# kill_all.sh - Kill all remaining processes from HAL evaluation

echo "=== Killing HAL Evaluation Processes ==="

DOCKER_CMD_TIMEOUT="${DOCKER_CMD_TIMEOUT:-8}"
HAL_KILL_ALL_DOCKER="${HAL_KILL_ALL_DOCKER:-1}" # Default to aggressive cleanup

docker_cmd() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$DOCKER_CMD_TIMEOUT" docker "$@"
    else
        docker "$@"
    fi
}

resolve_docker_host() {
    if [ -z "${DOCKER_HOST:-}" ] && [ -n "${HAL_DOCKER_HOST:-}" ]; then
        export DOCKER_HOST="$HAL_DOCKER_HOST"
    fi
    if [ -z "${DOCKER_HOST:-}" ] && [ -S "/run/user/$UID/docker.sock" ]; then
        export DOCKER_HOST="unix:///run/user/$UID/docker.sock"
    fi
}

docker_ready=false
if command -v docker >/dev/null 2>&1; then
    resolve_docker_host
    if docker_cmd info >/dev/null 2>&1; then
        docker_ready=true
    else
        echo "WARN: Docker daemon not reachable; skipping Docker cleanup."
    fi
else
    echo "WARN: Docker CLI not found; skipping Docker cleanup."
fi

# Kill main runners first (SIGKILL to ensure they stop spawning)
echo "[1/6] Killing main runners..."
pkill -9 -f "run_all_benchmarks.sh" 2>/dev/null
pkill -9 -f "run_benchmark_with_data.sh" 2>/dev/null
pkill -9 -f "tail -f.*log" 2>/dev/null  # Kill log tailers

# Kill Python evaluation scripts
echo "[2/6] Killing python evaluation scripts..."
pkill -9 -f "eval_rubric.py" 2>/dev/null
pkill -9 -f "fixing_pipeline.py" 2>/dev/null
pkill -9 -f "claude_fixer" 2>/dev/null
pkill -9 -f "scripts/run_.*_fixes.py" 2>/dev/null
pkill -9 -f "run_.*_fixes.py" 2>/dev/null
pkill -9 -f "judge.py" 2>/dev/null
pkill -9 -f "pipeline.py" 2>/dev/null
pkill -9 -f "hal.cli" 2>/dev/null
pkill -9 -f "hal/cli.py" 2>/dev/null
pkill -9 -f "python -m hal.cli" 2>/dev/null

# Kill agent processes
echo "[3/6] Killing agent processes..."
pkill -9 -f "run_agent.py" 2>/dev/null
pkill -9 -f "scicode_tool_calling_agent" 2>/dev/null
pkill -9 -f "hal_generalist_agent" 2>/dev/null
pkill -9 -f "SWE-agent" 2>/dev/null
pkill -9 -f "assistantbench_browser_agent" 2>/dev/null
pkill -9 -f "colbench_example_agent" 2>/dev/null
pkill -9 -f "smolagents" 2>/dev/null

# Cleanup generic python args if they look like ours
pkill -9 -f "python.*-u scripts/" 2>/dev/null

# Kill Docker containers related to evaluation
echo "[4/6] Stopping Docker containers..."
if $docker_ready; then
    if [ "$HAL_KILL_ALL_DOCKER" = "1" ]; then
        echo "HAL_KILL_ALL_DOCKER=1: Removing all containers..."
        docker_cmd ps -aq | xargs -r docker rm -f 2>/dev/null
    else
        # Targeted cleanup
        TARGETS=(
            "name=agentrun"
            "name=agentpool"
            "name=agent-env"
            "name=agentpreflight"
            "name=hal"
            "name=benchmark"
            "ancestor=hal-agent-runner"
        )
        for t in "${TARGETS[@]}"; do
            docker_cmd ps -aq --filter "$t" | xargs -r docker rm -f 2>/dev/null
        done
    fi
fi

# Final sweep
echo "[5/6] Final process sweep..."
pkill -9 -f "hal-eval" 2>/dev/null
pkill -9 -f "hal_eval" 2>/dev/null

# Show remaining processes (for verification)
echo ""
echo "=== Remaining Docker Containers ==="
if $docker_ready; then
    docker_cmd ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" 2>/dev/null || echo "None"
else
    echo "Docker unavailable"
fi

echo ""
echo "=== Done ==="
