#!/bin/bash
# Docker cleanup for HAL benchmark - PRESERVES hal-agent-runner base image
# Safe to run on shared machines

echo "=========================================="
echo "HAL Benchmark Docker Cleanup"
echo "=========================================="
echo ""

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
    if docker info >/dev/null 2>&1; then
        docker_ready=true
    else
        echo "WARN: Docker daemon not reachable; skipping Docker cleanup."
    fi
else
    echo "WARN: Docker CLI not found; skipping Docker cleanup."
fi

if ! $docker_ready; then
    echo ""
    echo "=========================================="
    echo "AFTER cleanup:"
    df -h / | grep -v Filesystem
    echo "Docker unavailable"
    echo "=========================================="
    exit 0
fi

# Show before state
echo "BEFORE cleanup:"
df -h / | grep -v Filesystem
if command -v timeout >/dev/null 2>&1; then
    timeout 5 docker system df 2>/dev/null || echo "Docker info timed out"
else
    docker system df 2>/dev/null || echo "Docker info unavailable"
fi
echo ""

# Step 1: Kill benchmark containers (by name pattern)
echo "Step 1: Stopping benchmark containers..."
docker kill $(docker ps -q --filter "name=agentrun" --filter "name=agent-env" --filter "name=agentpreflight") 2>/dev/null || echo "  None running"

# Step 2: Remove all stopped containers created TODAY (your runs)
echo "Step 2: Removing today's stopped containers..."
TODAY=$(date +%Y-%m-%d)
CONTAINERS=$(docker ps -aq --filter "status=exited" --filter "status=created")
if [ -n "$CONTAINERS" ]; then
    # Only remove containers created today
    for c in $CONTAINERS; do
        CREATED=$(docker inspect --format '{{.Created}}' $c 2>/dev/null | cut -d'T' -f1)
        if [ "$CREATED" = "$TODAY" ]; then
            docker rm -f $c >/dev/null 2>&1
        fi
    done
    echo "  Removed today's containers"
else
    echo "  No containers to remove"
fi

# Step 3: Remove dangling images (safe - untagged)
echo "Step 3: Removing dangling images..."
docker image prune -f 2>/dev/null || true

# Step 4: Remove agent-env-* images (but KEEP hal-agent-runner!)
echo "Step 4: Removing agent-env-* images (keeping hal-agent-runner)..."
docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | grep "^agent-env-" | awk '{print $2}' | xargs -r docker rmi -f 2>/dev/null || echo "  None to remove"

# Step 5: Remove build cache
echo "Step 5: Removing build cache..."
docker builder prune -af 2>/dev/null || true

# Show after state
echo ""
echo "=========================================="
echo "AFTER cleanup:"
df -h / | grep -v Filesystem
if command -v timeout >/dev/null 2>&1; then
    timeout 5 docker system df 2>/dev/null || echo "Docker info timed out"
else
    docker system df 2>/dev/null || echo "Docker info unavailable"
fi
echo ""
echo "Images preserved:"
docker images --format "  {{.Repository}}:{{.Tag}} ({{.Size}})"
echo "=========================================="
