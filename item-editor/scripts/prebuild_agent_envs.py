#!/usr/bin/env python3
"""
Pre-build all agent-env Docker images using HAL's actual hash calculation.

This script parses constants DIRECTLY from HAL's docker_runner.py to ensure
hash calculations always match. No manual synchronization or HAL imports needed.
"""

import sys
import os
import re
import json
import hashlib
from pathlib import Path

import docker

# Path to HAL harness
HAL_HARNESS = Path(__file__).parent.parent / "hal-harness"
DOCKER_RUNNER_PY = HAL_HARNESS / "hal" / "utils" / "docker_runner.py"


def parse_hal_constants():
    """Parse constants directly from docker_runner.py without importing.

    This avoids dependency chain issues (pydantic, etc.) while ensuring
    we always use the exact same values as HAL.
    """
    content = DOCKER_RUNNER_PY.read_text()

    # Parse DOCKER_IMAGE_NAME = "..."
    match = re.search(r'DOCKER_IMAGE_NAME\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find DOCKER_IMAGE_NAME in docker_runner.py")
    docker_image_name = match.group(1)

    # Parse AGENT_ENV_PYTHON_VERSION = "..."
    match = re.search(r'AGENT_ENV_PYTHON_VERSION\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find AGENT_ENV_PYTHON_VERSION in docker_runner.py")
    python_version = match.group(1)

    # Parse AGENT_ENV_TEMPLATE_VERSION = "..."
    match = re.search(r'AGENT_ENV_TEMPLATE_VERSION\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find AGENT_ENV_TEMPLATE_VERSION in docker_runner.py")
    template_version = match.group(1)

    return docker_image_name, python_version, template_version


# Parse constants from HAL's docker_runner.py - always in sync!
DOCKER_IMAGE_NAME, AGENT_ENV_PYTHON_VERSION, AGENT_ENV_TEMPLATE_VERSION = parse_hal_constants()

# Determine repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent

def get_agents_from_config(benchmark: str) -> set:
    """Extract unique agent directory names from benchmark config."""
    config_path = REPO_ROOT / "model_configs" / f"model_to_baseline_{benchmark}.json"
    if not config_path.exists():
        print(f"  [WARN] Config not found: {config_path}")
        return set()
    
    try:
        data = json.loads(config_path.read_text())
        agents = set()
        for key, entry in data.items():
            if key.startswith("_"): continue
            agent_dir = entry.get("agent_dir")
            if agent_dir:
                # agent_dir is usually like "hal-harness/agents/sab_example_agent"
                # we just want "sab_example_agent"
                agent_name = Path(agent_dir).name
                agents.add(agent_name)
        return agents
    except Exception as e:
        print(f"  [WARN] Failed to parse {config_path}: {e}")
        return set()

def image_exists(docker_client, tag: str) -> bool:
    """Check if Docker image exists."""
    try:
        docker_client.images.get(tag)
        return True
    except docker.errors.ImageNotFound:
        return False


def compute_requirements_hash(docker_client, requirements_path: str) -> str:
    """Compute requirements hash exactly as HAL does in _requirements_hash().

    Hash = sha256(requirements.txt + base_image_id + recipe)[:16]

    This must match HAL's docker_runner.py _requirements_hash() exactly!
    """
    req_bytes = Path(requirements_path).read_bytes()

    # Get base image ID (matches HAL's logic)
    try:
        base_image_id = docker_client.images.get(DOCKER_IMAGE_NAME).id.encode("utf-8")
    except Exception:
        base_image_id = b"unknown-base-image"

    # Recipe string (matches HAL's _requirements_hash exactly)
    recipe = (
        f"template={AGENT_ENV_TEMPLATE_VERSION}\n"
        f"python={AGENT_ENV_PYTHON_VERSION}\n"
        "weave=0.51.41\n"
        "wandb=0.17.9\n"
    ).encode("utf-8")

    return hashlib.sha256(req_bytes + b"\n" + base_image_id + b"\n" + recipe).hexdigest()[:16]


def build_agent_env(docker_client, agent_dir: str, agents_path: Path) -> bool:
    """Build agent-env image for a specific agent using HAL's exact hash calculation."""
    req_file = agents_path / agent_dir / "requirements.txt"

    if not req_file.exists():
        print(f"  [SKIP] {agent_dir} - no requirements.txt")
        return True

    # Use HAL's actual hash calculation method
    req_hash = compute_requirements_hash(docker_client, str(req_file))
    tag = f"hal-agent-runner:agent-env-{req_hash}"

    if image_exists(docker_client, tag):
        print(f"  [OK] {agent_dir} ({tag})")
        return True

    print(f"  [BUILD] {agent_dir} -> {tag}")
    num_packages = sum(1 for line in req_file.open() if line.strip() and not line.startswith('#'))
    print(f"    Installing {num_packages} packages...")
    sys.stdout.flush()

    # Create Dockerfile content - must match HAL's _ensure_prepared_image exactly
    # Using mamba for template v7+
    dockerfile = f"""
FROM {DOCKER_IMAGE_NAME}

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true && \\
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

RUN mamba create -y -n agent_env python={AGENT_ENV_PYTHON_VERSION} && \\
    conda run -n agent_env python -m pip install -U pip

RUN conda run -n agent_env pip install matplotlib numpy pandas scipy scikit-learn seaborn PyPDF2 xgboost ddgs beautifulsoup4 lxml

COPY requirements.txt /tmp/requirements.txt
RUN conda run -n agent_env pip install -r /tmp/requirements.txt && \\
    conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9

WORKDIR /workspace
"""

    # Write temporary Dockerfile
    build_context = agents_path / agent_dir
    dockerfile_path = build_context / "Dockerfile.agent-env"
    dockerfile_path.write_text(dockerfile)

    try:
        # Use low-level API for streaming output
        api_client = docker_client.api

        print(f"    Building (this may take 10-15 minutes)...")
        sys.stdout.flush()

        # Build with streaming
        build_stream = api_client.build(
            path=str(build_context),
            dockerfile="Dockerfile.agent-env",
            tag=tag,
            rm=True,
            decode=True,
        )

        last_step = ""
        for chunk in build_stream:
            if 'stream' in chunk:
                line = chunk['stream'].strip()
                if line:
                    # Show step progress
                    if line.startswith('Step '):
                        last_step = line
                        print(f"    {line}")
                        sys.stdout.flush()
                    # Show pip install progress
                    elif 'Successfully installed' in line:
                        print(f"    {line[:80]}...")
                        sys.stdout.flush()
                    elif 'Collecting' in line and 'from' not in line.lower():
                        # Show package being collected (first 60 chars)
                        pkg = line.replace('Collecting ', '').split()[0]
                        print(f"      Installing: {pkg[:50]}", end='\r')
                        sys.stdout.flush()
            elif 'error' in chunk:
                print(f"    ERROR: {chunk['error']}")
                sys.stdout.flush()
                return False

        print(f"\n  [DONE] {agent_dir}")
        sys.stdout.flush()
        return True

    except docker.errors.BuildError as e:
        print(f"\n  [FAIL] {agent_dir}: {e}")
        sys.stdout.flush()
        return False
    except Exception as e:
        print(f"\n  [FAIL] {agent_dir}: {type(e).__name__}: {e}")
        sys.stdout.flush()
        return False
    finally:
        dockerfile_path.unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*", help="Benchmarks to build images for (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("Pre-building Agent Environment Images")
    print("=" * 60)
    print()
    print(f"Using HAL constants (parsed from hal/utils/docker_runner.py):")
    print(f"  DOCKER_IMAGE_NAME: {DOCKER_IMAGE_NAME}")
    print(f"  AGENT_ENV_TEMPLATE_VERSION: {AGENT_ENV_TEMPLATE_VERSION}")
    print(f"  AGENT_ENV_PYTHON_VERSION: {AGENT_ENV_PYTHON_VERSION}")
    print()
    sys.stdout.flush()

    docker_client = docker.from_env()
    agents_path = HAL_HARNESS / "agents"

    # Check base image
    print("Step 1: Checking base image...")
    sys.stdout.flush()
    if not image_exists(docker_client, DOCKER_IMAGE_NAME):
        print(f"  [BUILD] Building {DOCKER_IMAGE_NAME}...")
        sys.stdout.flush()
        dockerfile_path = HAL_HARNESS / "hal" / "utils" / "docker"

        # Stream base image build too
        api_client = docker_client.api
        for chunk in api_client.build(path=str(dockerfile_path), tag=DOCKER_IMAGE_NAME, rm=True, decode=True):
            if 'stream' in chunk:
                line = chunk['stream'].strip()
                if line.startswith('Step '):
                    print(f"    {line}")
                    sys.stdout.flush()

        print(f"  [DONE] Base image built")
    else:
        print(f"  [OK] {DOCKER_IMAGE_NAME} exists")
    print()
    sys.stdout.flush()

    # Collect unique agents
    all_agents = set()
    
    # Filter benchmarks if provided
    benchmarks_to_process = []
    if args.benchmarks:
        for b in args.benchmarks:
            # Handle comma-separated list
            for part in b.split(','):
                bench = part.strip()
                if bench:
                    benchmarks_to_process.append(bench)
    
    if not benchmarks_to_process:
        # Default to all known benchmarks in model_configs
        for f in (REPO_ROOT / "model_configs").glob("model_to_baseline_*.json"):
            bench = f.stem.replace("model_to_baseline_", "")
            benchmarks_to_process.append(bench)
        
    print(f"Benchmarks to process: {', '.join(benchmarks_to_process)}")

    for bench in benchmarks_to_process:
        agents = get_agents_from_config(bench)
        if agents:
            all_agents.update(agents)
        else:
            print(f"  [WARN] No agents found for benchmark: {bench}")

    print("Step 2: Building agent-env images...")
    print(f"Agents to build: {', '.join(sorted(all_agents))}")
    print()
    sys.stdout.flush()

    failed = 0
    for agent in sorted(all_agents):
        if not build_agent_env(docker_client, agent, agents_path):
            failed += 1

    print()
    print("=" * 60)
    print("Summary:")
    try:
        images = docker_client.images.list()
    except docker.errors.ImageNotFound as e:
        print(f"  [WARN] Failed to list images: {e}")
        images = []
    except Exception as e:
        print(f"  [WARN] Failed to list images: {type(e).__name__}: {e}")
        images = []
    for img in images:
        try:
            tags = img.tags or []
        except docker.errors.ImageNotFound:
            continue
        except Exception:
            continue
        for tag in tags:
            if tag.startswith("hal-agent-runner:"):
                try:
                    size_bytes = img.attrs.get("Size")
                except docker.errors.ImageNotFound:
                    size_bytes = None
                except Exception:
                    size_bytes = None
                if size_bytes:
                    size_mb = size_bytes / 1024 / 1024
                    print(f"  {tag} ({size_mb:.0f}MB)")
                else:
                    print(f"  {tag}")
    print()

    if failed == 0:
        print(f"All {len(all_agents)} agent-env images ready!")
        print()
        print("You can now run benchmarks with full parallelism:")
        print("  ./run_all_benchmarks.sh moon5_ 20")
    else:
        print(f"WARNING: {failed} image(s) failed to build")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
