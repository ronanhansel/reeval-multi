#!/bin/bash
#
# Comprehensive Docker Image Prebuild Script
# Builds ALL Docker images needed for benchmark execution BEFORE any parallel runs.
#
# This ensures 800+ parallel processes don't race to build images.
#
# Images built:
# 1. hal-agent-runner:latest (base agent runner)
# 2. hal-agent-runner:agent-env-* (per-agent environments)
# 3. sab.base.x86_64:latest (ScienceAgentBench base)
# 4. SciCode evaluation image (if needed)
#
# Usage: ./prebuild_all_images.sh [--force]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${HAL_REPO_ROOT:-}" ]; then
    REPO_ROOT="$HAL_REPO_ROOT"
elif [ -d "$SCRIPT_DIR/../hal-harness" ] || [ -d "$SCRIPT_DIR/../scripts" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    REPO_ROOT="$SCRIPT_DIR"
fi
HAL_HARNESS="$REPO_ROOT/hal-harness"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
FORCE_REBUILD=false
BENCHMARKS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_REBUILD=true
            shift
            ;;
        *)
            BENCHMARKS="$BENCHMARKS $1"
            shift
            ;;
    esac
done

if [[ "$FORCE_REBUILD" == "true" ]]; then
    echo -e "${YELLOW}[prebuild] Force rebuild enabled - will rebuild ALL images${NC}"
fi

if [[ -n "$BENCHMARKS" ]]; then
    echo -e "${BLUE}Benchmarks:${NC} $BENCHMARKS"
fi

# Retry function for transient failures
retry() {
    local max_attempts=$1
    local delay=$2
    local cmd="${@:3}"
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if eval "$cmd"; then
            return 0
        fi
        echo -e "${YELLOW}[retry] Attempt $attempt/$max_attempts failed, waiting ${delay}s...${NC}"
        sleep $delay
        ((attempt++))
    done
    return 1
}

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}      COMPREHENSIVE DOCKER IMAGE PREBUILD${NC}"
echo -e "${CYAN}================================================================${NC}"
echo -e "${BLUE}Script Dir:${NC} $SCRIPT_DIR"
echo -e "${BLUE}HAL Harness:${NC} $HAL_HARNESS"
echo -e "${BLUE}Force Rebuild:${NC} $FORCE_REBUILD"
echo -e "${CYAN}================================================================${NC}"
echo ""

# Track failures
FAILURES=0

# =============================================================================
# STEP 1: Build hal-agent-runner:latest (base image for all agents)
# =============================================================================
echo -e "${CYAN}[Step 1/4] Building hal-agent-runner:latest (base image)${NC}"

build_base_runner() {
    local image_name="hal-agent-runner:latest"
    local dockerfile_dir="$HAL_HARNESS/hal/utils/docker"

    if [[ "$FORCE_REBUILD" == "false" ]] && docker images -q "$image_name" 2>/dev/null | grep -q .; then
        echo -e "${GREEN}  [OK] $image_name already exists${NC}"
        return 0
    fi

    echo -e "${YELLOW}  [BUILD] Building $image_name...${NC}"
    if docker build -t "$image_name" "$dockerfile_dir" 2>&1 | tail -20; then
        echo -e "${GREEN}  [DONE] $image_name built successfully${NC}"
        return 0
    else
        echo -e "${RED}  [FAIL] Failed to build $image_name${NC}"
        return 1
    fi
}

if ! retry 3 10 build_base_runner; then
    echo -e "${RED}[FATAL] Could not build base image after 3 attempts${NC}"
    ((FAILURES++))
fi
echo ""

# =============================================================================
# STEP 2: Build agent-env images (per-agent environments)
# Uses prebuild_agent_envs.py which reads constants directly from HAL's docker_runner.py
# This ensures hash calculations ALWAYS match HAL - no manual synchronization needed!
# =============================================================================
echo -e "${CYAN}[Step 2/4] Building agent-env images (via prebuild_agent_envs.py)${NC}"

build_agent_envs() {
    # Optional: Call Python script which reads constants directly from HAL's docker_runner.py
    # This ensures hash calculations always match - no hardcoded values here!
    if [ -f "$SCRIPT_DIR/../scripts/prebuild_agent_envs.py" ]; then
        if python3 "$SCRIPT_DIR/../scripts/prebuild_agent_envs.py" $BENCHMARKS; then
            return 0
        else
            return 1
        fi
    else
        echo -e "${YELLOW}  [SKIP] prebuild_agent_envs.py not found - agent-env images will be built on demand${NC}"
        return 0
    fi
}

if ! retry 2 10 build_agent_envs; then
    echo -e "${RED}  [ERROR] Failed to build agent-env images after retries${NC}"
    ((FAILURES++))
fi
echo ""

# =============================================================================
# STEP 3: Build sab.base.x86_64:latest (ScienceAgentBench base image)
# =============================================================================
if [[ -z "$BENCHMARKS" ]] || [[ "$BENCHMARKS" == *"scienceagentbench"* ]]; then
    echo -e "${CYAN}[Step 3/4] Building ScienceAgentBench base image${NC}"

    build_sab_base() {
        local image_name="sab.base.x86_64:latest"

        if [[ "$FORCE_REBUILD" == "false" ]] && docker images -q "$image_name" 2>/dev/null | grep -q .; then
            echo -e "${GREEN}  [OK] $image_name already exists${NC}"
            return 0
        fi

        echo -e "${YELLOW}  [BUILD] Building $image_name (this may take 5-10 minutes)...${NC}"

        # Use Python to build via the official dockerfiles.py
        python3 << PYTHON_SCRIPT
import docker
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '$HAL_HARNESS/hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/evaluation/harness')
from dockerfiles import get_dockerfile_base

client = docker.from_env(timeout=600)

# Get the dockerfile content
dockerfile = get_dockerfile_base('linux/x86_64', 'x86_64')

with tempfile.TemporaryDirectory() as tmpdir:
    dockerfile_path = Path(tmpdir) / 'Dockerfile'
    dockerfile_path.write_text(dockerfile)

    print('Building sab.base.x86_64:latest...')
    print('Dockerfile content:')
    print(dockerfile[:500] + '...')

    try:
        image, logs = client.images.build(
            path=tmpdir,
            tag='sab.base.x86_64:latest',
            platform='linux/x86_64',
            rm=True,
            timeout=600,
            nocache=False  # Use cache for faster rebuilds
        )
        print(f'SUCCESS: Built {image.tags}')
    except docker.errors.BuildError as e:
        print(f'BUILD ERROR: {e}')
        # Print build log
        for log in e.build_log:
            if 'stream' in log:
                print(log['stream'], end='')
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: {e}')
        sys.exit(1)
PYTHON_SCRIPT

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  [DONE] $image_name built successfully${NC}"
            return 0
        else
            echo -e "${RED}  [FAIL] Failed to build $image_name${NC}"
            return 1
        fi
    }

    if ! retry 3 30 build_sab_base; then
        echo -e "${RED}[ERROR] Failed to build SAB base image after 3 attempts${NC}"
        ((FAILURES++))
    fi
    echo ""
else
    echo -e "${YELLOW}[Step 3/4] Skipping ScienceAgentBench base image (not requested)${NC}"
    echo ""
fi

# =============================================================================
# STEP 3.5: Build scicode-eval:latest (SciCode evaluation image)
# =============================================================================
if [[ -z "$BENCHMARKS" ]] || [[ "$BENCHMARKS" == *"scicode"* ]]; then
    echo -e "${CYAN}[Step 3.5/4] Building SciCode evaluation image${NC}"

    build_scicode_eval() {
        local image_name="scicode-eval:latest"

        if [[ "$FORCE_REBUILD" == "false" ]] && docker images -q "$image_name" 2>/dev/null | grep -q .; then
            echo -e "${GREEN}  [OK] $image_name already exists${NC}"
            return 0
        fi

        echo -e "${YELLOW}  [BUILD] Building $image_name...${NC}"
        
        if python3 "$SCRIPT_DIR/build_scicode_image.py"; then
             echo -e "${GREEN}  [DONE] $image_name built successfully${NC}"
             return 0
        else
             echo -e "${RED}  [FAIL] Failed to build $image_name${NC}"
             return 1
        fi
    }

    if ! retry 3 30 build_scicode_eval; then
        echo -e "${RED}[ERROR] Failed to build SciCode eval image after 3 attempts${NC}"
        ((FAILURES++))
    fi
    echo ""
else
    echo -e "${YELLOW}[Step 3.5/4] Skipping SciCode eval image (not requested)${NC}"
    echo ""
fi

# =============================================================================
# STEP 4: Pull required base images
# =============================================================================
echo -e "${CYAN}[Step 4/4] Pulling base images${NC}"

pull_base_images() {
    local images=("ubuntu:22.04" "python:3.11-slim")

    for img in "${images[@]}"; do
        if docker images -q "$img" 2>/dev/null | grep -q .; then
            echo -e "${GREEN}  [OK] $img already exists${NC}"
        else
            echo -e "${YELLOW}  [PULL] $img${NC}"
            docker pull "$img" || echo -e "${YELLOW}  [WARN] Could not pull $img${NC}"
        fi
    done
}

pull_base_images
echo ""

# =============================================================================
# VERIFICATION
# =============================================================================
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}      VERIFICATION${NC}"
echo -e "${CYAN}================================================================${NC}"

verify_image() {
    local pattern="$1"
    local desc="$2"
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -qE "$pattern"; then
        echo -e "${GREEN}  [OK] $desc${NC}"
        return 0
    else
        echo -e "${RED}  [MISSING] $desc${NC}"
        return 1
    fi
}

echo ""
echo "Checking required images:"
verify_image "^hal-agent-runner:latest$" "Base runner image" || ((FAILURES++))
verify_image "^hal-agent-runner:agent-env-" "Agent environment images" || ((FAILURES++))

if [[ -z "$BENCHMARKS" ]] || [[ "$BENCHMARKS" == *"scienceagentbench"* ]]; then
    verify_image "^sab\.base\.x86_64:latest$" "ScienceAgentBench base" || ((FAILURES++))
fi

verify_image "^ubuntu:22\.04$" "Ubuntu 22.04 base" || true  # Not critical

echo ""
echo "All Docker images:"
docker images --format "  {{.Repository}}:{{.Tag}} ({{.Size}})" | grep -E "hal-agent-runner|sab\." | sort
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "${CYAN}================================================================${NC}"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}      SUCCESS: All images ready!${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo "You can now run benchmarks with full parallelism:"
    echo "  ./run_all_benchmarks.sh moon2_ 20"
    echo ""
    exit 0
else
    echo -e "${RED}      FAILED: $FAILURES image(s) could not be built${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo "Please check the errors above and retry."
    echo "Common issues:"
    echo "  - Network connectivity (DNS resolution)"
    echo "  - Disk space"
    echo "  - Docker daemon not running"
    echo ""
    exit 1
fi
