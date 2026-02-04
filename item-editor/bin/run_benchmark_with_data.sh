#!/bin/bash
# Complete wrapper to run benchmarks using /Data for temporary storage
# This prevents filling up the root partition
# Also pre-builds agent-env Docker images to avoid build contention

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${HAL_REPO_ROOT:-}" ]; then
    REPO_ROOT="$HAL_REPO_ROOT"
elif [ -d "$SCRIPT_DIR/../hal-harness" ] || [ -d "$SCRIPT_DIR/../scripts" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_ROOT="$SCRIPT_DIR"
fi
WORKDIR="${HAL_WORKDIR:-$REPO_ROOT}"
HAL_HARNESS="$REPO_ROOT/hal-harness"

# Ensure Docker runner loads the Azure/TRAPI environment when present.
if [ -z "${HAL_DOTENV_PATH:-}" ] && [ -f "$HAL_HARNESS/.env" ]; then
    export HAL_DOTENV_PATH="$HAL_HARNESS/.env"
fi

# Skip MSAL preflight by default to handle high-parallelism (e.g. 400+ processes)
# without bottlenecking on initial token verification.
export HAL_SKIP_MSAL_PREFLIGHT="${HAL_SKIP_MSAL_PREFLIGHT:-true}"

# Optional rootless Docker mode (set ROOTLESS=TRUE)
is_truthy() {
    case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

if is_truthy "${ROOTLESS:-}"; then
    ROOTLESS_SOCKET="${ROOTLESS_SOCKET:-/run/user/$UID/docker.sock}"
    export DOCKER_HOST="${DOCKER_HOST:-unix://$ROOTLESS_SOCKET}"
    export HAL_DOCKER_HOST="${HAL_DOCKER_HOST:-$DOCKER_HOST}"
    echo "[Docker] Rootless mode enabled: $DOCKER_HOST"
fi

# =============================================================================
# Dynamic storage detection - works on any machine
# =============================================================================
# Priority order:
# 1. DATA_PATH environment variable (explicit override)
# 2. HAL_DATA_ROOT environment variable (explicit override)
# 3. /Data/home/${USER} if exists and writable
# 4. /Data if exists and writable
# 5. Repository's .hal_data directory (always works, uses repo storage)

detect_data_root() {
    # Check if explicitly set
    if [ -n "${DATA_PATH:-}" ]; then
        if [ -d "$DATA_PATH" ] && [ -w "$DATA_PATH" ]; then
            echo "$DATA_PATH"
            return
        else
            echo "WARN: DATA_PATH=$DATA_PATH is not writable; ignoring." >&2
        fi
    fi

    if [ -n "${HAL_DATA_ROOT:-}" ]; then
        if [ -d "$HAL_DATA_ROOT" ] && [ -w "$HAL_DATA_ROOT" ]; then
            echo "$HAL_DATA_ROOT"
            return
        else
            echo "WARN: HAL_DATA_ROOT=$HAL_DATA_ROOT is not writable; ignoring." >&2
        fi
    fi

    # Try /Data/home/${USER}
    local data_home="/Data/home/${USER}"
    if [ -d "$data_home" ] && [ -w "$data_home" ]; then
        echo "$data_home"
        return
    fi

    # Try /Data
    if [ -d "/Data" ] && [ -w "/Data" ]; then
        echo "/Data"
        return
    fi

    # Fall back to repository-local storage
    local local_data="$REPO_ROOT/.hal_data"
    mkdir -p "$local_data" 2>/dev/null || true
    echo "$local_data"
}

DATA_ROOT="$(detect_data_root)"
DATA_NAMESPACE="${HAL_DATA_NAMESPACE:-$USER}"

# If using repo-local storage, simplify the path structure
if [[ "$DATA_ROOT" == "$REPO_ROOT/.hal_data" ]]; then
    DATA_RUN_ROOT="${HAL_DATA_RUN_ROOT:-$DATA_ROOT}"
else
    DATA_RUN_ROOT="${HAL_DATA_RUN_ROOT:-$DATA_ROOT/hal_runs/$DATA_NAMESPACE/$(basename "$REPO_ROOT")}"
fi

echo "[Storage] Using data root: $DATA_ROOT"
echo "[Storage] Run root: $DATA_RUN_ROOT"

# Set working directory
cd "$WORKDIR"

# Configure all temp directories to use /Data
export TMPDIR="${HAL_TMPDIR:-$DATA_RUN_ROOT/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHON_TEMPDIR="$TMPDIR"
export DOCKER_TMPDIR="${HAL_DOCKER_TMPDIR:-$TMPDIR/docker}"
export HAL_RESULTS_DIR="${HAL_RESULTS_DIR:-$DATA_RUN_ROOT/results}"
export HAL_TRACES_DIR="${HAL_TRACES_DIR:-$DATA_RUN_ROOT/traces}"
export HAL_TMP_DIR="${HAL_TMP_DIR:-$DATA_RUN_ROOT/tmp}"
if [ -n "${DATA_PATH:-}" ] && [ -z "${XDG_CACHE_HOME:-}" ]; then
    export XDG_CACHE_HOME="$DATA_RUN_ROOT/.cache"
fi

# Ensure temp directory exists
mkdir -p "$TMPDIR" "$DOCKER_TMPDIR" "$HAL_RESULTS_DIR" "$HAL_TRACES_DIR" "$HAL_TMP_DIR"
if [ -n "${XDG_CACHE_HOME:-}" ]; then
    mkdir -p "$XDG_CACHE_HOME"
fi

DATA_PATH_ENFORCE=false
if [ -n "${DATA_PATH:-}" ]; then
    DATA_PATH_ENFORCE=true
fi

link_dir() {
    local name="$1"
    local src="$WORKDIR/$name"
    local target="$DATA_RUN_ROOT/$name"
    if [ -L "$src" ]; then
        if [ ! -e "$src" ]; then
            echo "WARN: $src is a broken symlink. Re-linking to $target."
            rm -f "$src"
        else
            return
        fi
    fi
    if [ -e "$src" ]; then
        if $DATA_PATH_ENFORCE || is_truthy "${HAL_MIGRATE_DATA_DIRS:-}"; then
            local stamp
            stamp=$(date +%Y%m%d_%H%M%S)
            if [ ! -d "$src" ]; then
                echo "WARN: $src exists and is not a directory. Skipping migrate."
                return
            fi
            if [ -e "$target" ] && [ ! -d "$target" ]; then
                echo "WARN: $target exists and is not a directory. Skipping migrate."
                return
            fi
            if [ -e "$target" ]; then
                mkdir -p "$target"
                local migrated="$target/_migrated_$stamp"
                echo "Moving existing $src to $migrated"
                if ! mv "$src" "$migrated"; then
                    echo "WARN: Failed to move $src to $migrated"
                    return
                fi
            else
                echo "Moving existing $src to $target"
                if ! mv "$src" "$target"; then
                    echo "WARN: Failed to move $src to $target"
                    return
                fi
            fi
        else
            echo "WARN: $src exists and is not a symlink. Move it to $target or delete it to enable linking."
            return
        fi
    fi
    mkdir -p "$target"
    ln -s "$target" "$src"
}

if [ "${HAL_LINK_DATA_DIRS:-1}" != "0" ]; then
    link_dir "results"
    link_dir "traces"
    link_dir ".tmp"
    link_dir "log"
    link_dir "logs"
    link_dir "output"
fi

# =============================================================================
# Agent-env Docker image pre-building
# =============================================================================

# Map benchmarks to their required agent directories
declare -A BENCHMARK_AGENTS
BENCHMARK_AGENTS[scicode]="hal_generalist_agent scicode_tool_calling_agent"
BENCHMARK_AGENTS[scienceagentbench]="hal_generalist_agent sab_example_agent"
BENCHMARK_AGENTS[corebench]="hal_generalist_agent core_agent"
BENCHMARK_AGENTS[colbench]="hal_generalist_agent colbench_example_agent"

# Compute agent-env image tag from requirements.txt hash
get_agent_env_tag() {
    local agent_dir="$1"
    local req_file="$HAL_HARNESS/agents/$agent_dir/requirements.txt"
    if [ -f "$req_file" ]; then
        local hash=$(md5sum "$req_file" | cut -c1-12)
        echo "hal-agent-runner:agent-env-$hash"
    else
        echo ""
    fi
}

# Check if image exists
image_exists() {
    local tag="$1"
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${tag}$"
}

# Build agent-env image for a specific agent
build_agent_env() {
    local agent_dir="$1"
    local tag=$(get_agent_env_tag "$agent_dir")
    local req_file="$HAL_HARNESS/agents/$agent_dir/requirements.txt"

    if [ -z "$tag" ]; then
        echo "  [SKIP] $agent_dir - no requirements.txt"
        return 0
    fi

    if image_exists "$tag"; then
        echo "  [OK] $agent_dir - image exists ($tag)"
        return 0
    fi

    echo "  [BUILD] $agent_dir - building $tag..."

    # Create a temporary Dockerfile for agent-env
    local tmp_dockerfile=$(mktemp)
    cat > "$tmp_dockerfile" << 'DOCKERFILE'
ARG BASE_IMAGE=hal-agent-runner:latest
FROM ${BASE_IMAGE}

# Accept conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create agent environment (using mamba for speed)
RUN mamba create -y -n agent_env python=3.11 && \
    conda run -n agent_env python -m pip install -U pip

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n agent_env pip install -r /tmp/requirements.txt && \
    conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9

WORKDIR /workspace
DOCKERFILE

    # Build the image
    local build_context="$HAL_HARNESS/agents/$agent_dir"
    if docker build -t "$tag" -f "$tmp_dockerfile" "$build_context" > /dev/null 2>&1; then
        echo "  [DONE] $agent_dir - built successfully"
        rm -f "$tmp_dockerfile"
        return 0
    else
        echo "  [FAIL] $agent_dir - build failed, will retry at runtime"
        rm -f "$tmp_dockerfile"
        return 1
    fi
}

# Pre-build all required agent-env images for detected benchmarks
prebuild_agent_envs() {
    local benchmarks_to_build=""

    # Detect which benchmarks are in the command line
    for arg in "$@"; do
        case "$arg" in
            scicode|scienceagentbench|corebench|colbench)
                benchmarks_to_build="$benchmarks_to_build $arg"
                ;;
        esac
    done

    if [ -z "$benchmarks_to_build" ]; then
        return 0
    fi

    echo "=========================================="
    echo "Pre-building Agent Environment Images"
    echo "=========================================="

    # First ensure base image exists
    if ! image_exists "hal-agent-runner:latest"; then
        echo "Building base image hal-agent-runner:latest..."
        docker build -t hal-agent-runner:latest "$HAL_HARNESS/hal/utils/docker/" || {
            echo "ERROR: Failed to build base image"
            exit 1
        }
    else
        echo "Base image hal-agent-runner:latest exists"
    fi

    # Collect unique agents needed
    declare -A agents_needed
    for bench in $benchmarks_to_build; do
        for agent in ${BENCHMARK_AGENTS[$bench]}; do
            agents_needed[$agent]=1
        done
    done

    echo ""
    echo "Agents needed: ${!agents_needed[@]}"
    echo ""

    # Build each agent-env
    local all_ok=true
    for agent in "${!agents_needed[@]}"; do
        build_agent_env "$agent" || all_ok=false
    done

    echo ""
    if $all_ok; then
        echo "All agent-env images ready!"
    else
        echo "Some images failed to pre-build (will retry at runtime)"
    fi
    echo "=========================================="
    echo ""
}

# =============================================================================
# Main script
# =============================================================================

# Show current disk usage
echo "=========================================="
echo "Disk Usage Status:"
echo "=========================================="
echo "Root partition:"
df -h / | tail -1
echo ""
echo "Data storage partition ($DATA_ROOT):"
df -h "$DATA_ROOT" 2>/dev/null | tail -1 || echo "(using same as root)"
echo ""
echo "Temp directory: $TMPDIR"
echo "Available for temp files: $(df -h "$TMPDIR" 2>/dev/null | tail -1 | awk '{print $4}' || echo 'N/A')"
echo ""
echo "Docker usage:"
timeout 5s docker system df --format "table {{.Type}}\t{{.Size}}\t{{.Reclaimable}}" 2>/dev/null || echo "(docker df timed out or not available)"
echo "=========================================="
echo ""

# Check if we have enough space on root
ROOT_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
ROOT_CHECK_BYPASS=false
if is_truthy "${HAL_SKIP_ROOT_CHECK:-}"; then
    ROOT_CHECK_BYPASS=true
elif is_truthy "${ROOTLESS:-}" && [ "$DATA_ROOT" != "/" ]; then
    ROOT_CHECK_BYPASS=true
fi

if [ $ROOT_USAGE -ge 95 ]; then
    echo -e "\n[FATAL] STORAGE FULL WARNING" >&2
    echo "Root partition is ${ROOT_USAGE}% full! (Threshold: 95%)" >&2
    echo "This may cause Docker or the system to crash." >&2
    echo "Consider running ./cleanup_docker.sh first." >&2
    echo "" >&2
    if $ROOT_CHECK_BYPASS; then
        echo "Continuing despite full root (HAL_SKIP_ROOT_CHECK/ROOTLESS enabled)." >&2
        echo "" >&2
    elif [ -t 0 ]; then
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted by user. Please run ./cleanup_docker.sh to free space." >&2
            exit 1
        fi
    else
        echo "ABORTED: Non-interactive shell detected and storage is critical." >&2
        echo "To override, pass --skip-root-check to the runner script." >&2
        exit 1
    fi
fi

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Example:"
    echo "  $0 python scripts/run_benchmark_fixes.py --benchmark scicode --all-configs --all-tasks --prefix scicode_moon1_ --docker --parallel-models 10 --parallel-tasks 10"
    echo ""
    echo "Supported benchmarks (with auto pre-build):"
    echo "  scicode, scienceagentbench, corebench, colbench"
    echo ""
    exit 1
fi

# Pre-build agent-env images if running with --docker
if echo "$@" | grep -q "\-\-docker"; then
    # Use Python script for correct HAL hash calculation
    if [ -f "$SCRIPT_DIR/../scripts/prebuild_agent_envs.py" ]; then
        echo "Pre-building agent-env images (using HAL's hash calculation)..."
        python3 "$SCRIPT_DIR/../scripts/prebuild_agent_envs.py" 2>&1 | grep -E "^\[|^Step|^All|^WARNING|^Agents"
        echo ""
    else
        # Fallback to shell-based prebuild
        prebuild_agent_envs "$@"
    fi
fi

# Run the command
echo "Starting command with /Data temporary storage..."
echo "Command: $@"
echo ""

if [ "$1" = "python" ] && ! command -v python >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        echo "WARN: python not found; using python3"
        shift
        set -- python3 "$@"
    else
        echo "ERROR: python not found and python3 not available"
        exit 127
    fi
fi

exec "$@"
