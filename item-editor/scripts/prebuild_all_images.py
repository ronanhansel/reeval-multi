#!/usr/bin/env python3
"""
Comprehensive Docker Image Prebuild Script

Orchestrates the building of all required Docker images for HAL evaluation:
1. Base runner image (hal-agent-runner:latest)
2. Agent environment images (using HAL's hash-based naming)
3. Benchmark-specific images (ScienceAgentBench base, SciCode eval)
"""

import os
import subprocess
import sys
import argparse
import tempfile
import json
import hashlib
import re
from pathlib import Path
import docker
try:
    import toml
except ImportError:
    # Minimal fallback for toml if not installed
    toml = None

from hal_common import log, Colors, get_hal_harness_path, REPO_ROOT

# =============================================================================
# Helper Utilities
# =============================================================================

def run_command(cmd, cwd=None, capture_output=False, shell=False):
    try:
        result = subprocess.run(cmd, cwd=cwd, shell=shell, capture_output=capture_output, text=True, check=True)
        return True, result.stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr if capture_output else str(e)

def image_exists(docker_client, tag: str) -> bool:
    """Check if Docker image exists."""
    try:
        docker_client.images.get(tag)
        return True
    except:
        return False

# =============================================================================
# Base Runner Build
# =============================================================================

def build_base_runner(force=False):
    image_name = "hal-agent-runner:latest"
    harness_path = get_hal_harness_path()
    dockerfile_dir = harness_path / "hal" / "utils" / "docker"
    
    # Check if exists
    if not force:
        client = docker.from_env()
        if image_exists(client, image_name):
            log(f"[OK] {image_name} already exists", Colors.GREEN)
            return True

    log(f"[BUILD] Building {image_name}...", Colors.YELLOW)
    success, _ = run_command(["docker", "build", "-t", image_name, str(dockerfile_dir)])
    if success:
        log(f"[DONE] {image_name} built successfully", Colors.GREEN)
    else:
        log(f"[FAIL] Failed to build {image_name}: {_}", Colors.RED)
    return success

# =============================================================================
# Agent Environment Build Logic
# =============================================================================

def parse_hal_constants():
    """Parse constants directly from HAL's docker_runner.py."""
    harness_path = get_hal_harness_path()
    docker_runner_py = harness_path / "hal" / "utils" / "docker_runner.py"
    
    if not docker_runner_py.exists():
        log(f"Warning: {docker_runner_py} not found, using defaults", Colors.YELLOW)
        return "hal-agent-runner:latest", "3.10", "v7"

    content = docker_runner_py.read_text()
    
    def get_match(pattern, default):
        m = re.search(pattern, content)
        return m.group(1) if m else default

    docker_image_name = get_match(r'DOCKER_IMAGE_NAME\s*=\s*["\']([^"\']+)["\']', "hal-agent-runner:latest")
    python_version = get_match(r'AGENT_ENV_PYTHON_VERSION\s*=\s*["\']([^"\']+)["\']', "3.10")
    template_version = get_match(r'AGENT_ENV_TEMPLATE_VERSION\s*=\s*["\']([^"\']+)["\']', "v7")

    return docker_image_name, python_version, template_version


def compute_requirements_hash(requirements_path: Path, base_image_name: str, python_version: str, template_version: str) -> str:
    """Compute requirements hash exactly as HAL does."""
    req_bytes = requirements_path.read_bytes()
    client = docker.from_env()
    
    try:
        base_image_id = client.images.get(base_image_name).id.encode("utf-8")
    except:
        base_image_id = b"unknown-base-image"

    recipe = (
        f"template={template_version}\n"
        f"python={python_version}\n"
        "weave=0.51.41\n"
        "wandb=0.17.9\n"
    ).encode("utf-8")

    return hashlib.sha256(req_bytes + b"\n" + base_image_id + b"\n" + recipe).hexdigest()[:16]


def get_agents_from_configs(benchmarks: List[str]) -> set:
    """Extract agent names from benchmark configs."""
    agents = set()
    for bench in benchmarks:
        config_path = REPO_ROOT / "model_configs" / f"model_to_baseline_{bench}.json"
        if not config_path.exists(): continue
        try:
            data = json.loads(config_path.read_text())
            for key, entry in data.items():
                if key.startswith("_"): continue
                agent_dir = entry.get("agent_dir")
                if agent_dir:
                    agents.add(Path(agent_dir).name)
        except: pass
    return agents


def build_agent_envs(benchmarks=None, force=False):
    log("Building agent-env images...", Colors.CYAN)
    
    docker_image_name, python_version, template_version = parse_hal_constants()
    
    if not benchmarks:
        # Default to all known benchmarks
        benchmarks = [f.stem.replace("model_to_baseline_", "") 
                     for f in (REPO_ROOT / "model_configs").glob("model_to_baseline_*.json")]
    
    agents = get_agents_from_configs(benchmarks)
    log(f"Agents to process: {', '.join(sorted(agents))}", Colors.BLUE)
    
    harness_path = get_hal_harness_path()
    client = docker.from_env()
    failed = 0
    
    for agent_name in sorted(agents):
        agent_dir = harness_path / "agents" / agent_name
        req_file = agent_dir / "requirements.txt"
        
        if not req_file.exists():
            log(f"  [SKIP] {agent_name} - no requirements.txt", Colors.NC)
            continue
            
        req_hash = compute_requirements_hash(req_file, docker_image_name, python_version, template_version)
        tag = f"hal-agent-runner:agent-env-{req_hash}"
        
        if not force and image_exists(client, tag):
            log(f"  [OK] {agent_name} ({tag})", Colors.GREEN)
            continue
            
        log(f"  [BUILD] {agent_name} -> {tag}", Colors.YELLOW)
        
        dockerfile = f"""
FROM {docker_image_name}
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true && \\
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
RUN mamba create -y -n agent_env python={python_version} && \\
    conda run -n agent_env python -m pip install -U pip
RUN conda run -n agent_env pip install matplotlib numpy pandas scipy scikit-learn seaborn PyPDF2 xgboost ddgs beautifulsoup4 lxml
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n agent_env pip install -r /tmp/requirements.txt && \\
    conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9
WORKDIR /workspace
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "Dockerfile").write_text(dockerfile)
            import shutil
            shutil.copy2(req_file, tmp_path / "requirements.txt")
            
            try:
                cmd = ["docker", "build", "-t", tag, "."]
                success, output = run_command(cmd, cwd=tmpdir)
                if success:
                    log(f"  [DONE] {agent_name}", Colors.GREEN)
                else:
                    log(f"  [FAIL] {agent_name}: {output}", Colors.RED)
                    failed += 1
            except Exception as e:
                log(f"  [ERROR] {agent_name}: {e}", Colors.RED)
                failed += 1
                
    return failed == 0

# =============================================================================
# Benchmark-Specific Image Builds
# =============================================================================

def build_sab_base(force=False):
    image_name = "sab.base.x86_64:latest"
    if not force:
        client = docker.from_env()
        if image_exists(client, image_name):
            log(f"[OK] {image_name} already exists", Colors.GREEN)
            return True

    log(f"[BUILD] Building {image_name} (this may take 5-10 minutes)...", Colors.YELLOW)
    harness_path = get_hal_harness_path()
    
    # Python bridge to import SAB's dockerfile generator
    import sys
    sab_harness_path = harness_path / "hal/benchmarks/scienceagentbench/ScienceAgentBench_modified/evaluation/harness"
    if not sab_harness_path.exists():
        sab_harness_path = harness_path / "hal/benchmarks/scienceagentbench/ScienceAgentBench/evaluation/harness"
        
    sys.path.insert(0, str(sab_harness_path))
    try:
        from dockerfiles import get_dockerfile_base
        dockerfile = get_dockerfile_base('linux/x86_64', 'x86_64')
    except Exception as e:
        log(f"Failed to generate ScienceAgentBench Dockerfile: {e}", Colors.RED)
        return False
    finally:
        if str(sab_harness_path) in sys.path:
            sys.path.remove(str(sab_harness_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / 'Dockerfile').write_text(dockerfile)
        success, output = run_command(["docker", "build", "-t", image_name, "--platform", "linux/x86_64", "."], cwd=tmpdir)
        if success:
            log(f"[DONE] {image_name} built successfully", Colors.GREEN)
        else:
            log(f"[FAIL] {image_name} failed: {output}", Colors.RED)
        return success


def build_scicode_eval(force=False):
    image_name = "scicode-eval:latest"
    if not force:
        client = docker.from_env()
        if image_exists(client, image_name):
            log(f"[OK] {image_name} already exists", Colors.GREEN)
            return True

    log(f"[BUILD] Building {image_name}...", Colors.YELLOW)
    
    # Integrated logic from build_scicode_image.py
    pyproject_path = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "SciCode" / "pyproject.toml"
    
    if not pyproject_path.exists():
        log(f"Error: {pyproject_path} not found", Colors.RED)
        return False
        
    log(f"Reading dependencies from {pyproject_path}", Colors.BLUE)
    
    try:
        if toml:
            pyproject = toml.load(pyproject_path)
        else:
            # Fallback to crude regex parsing if toml not available
            content = pyproject_path.read_text()
            match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                deps_str = match.group(1)
                dependencies = [d.strip().strip('"').strip("'") for d in deps_str.split(',') if d.strip()]
            else:
                dependencies = []
            pyproject = {"project": {"dependencies": dependencies}}
            
        dependencies = pyproject.get("project", {}).get("dependencies", [])
        pip_packages = " ".join(dependencies)
        
        dockerfile_content = f"""
FROM python:3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir {pip_packages}
WORKDIR /app
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Dockerfile").write_text(dockerfile_content)
            success, output = run_command(["docker", "build", "-t", image_name, "."], cwd=tmpdir)
            if success:
                log(f"[DONE] {image_name} built successfully", Colors.GREEN)
            else:
                log(f"[FAIL] {image_name} failed: {output}", Colors.RED)
            return success
            
    except Exception as e:
        log(f"Error building SciCode image: {e}", Colors.RED)
        return False

# =============================================================================
# CoreBench Data Verification
# =============================================================================

def ensure_corebench_data():
    log("Verifying CoreBench datasets...", Colors.CYAN)
    harness_path = get_hal_harness_path()
    sys.path.insert(0, str(harness_path))
    
    try:
        from hal.benchmarks.corebench import CoreBenchHard
        # This will trigger __download_and_extract_capsule in __init__
        CoreBenchHard(agent_dir=".", config={})
        log("[OK] CoreBench data is ready", Colors.GREEN)
        return True
    except Exception as e:
        log(f"[FAIL] CoreBench data verification failed: {e}", Colors.RED)
        return False
    finally:
        if str(harness_path) in sys.path:
            sys.path.remove(str(harness_path))

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prebuild all Docker images for benchmarks.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of all images.")
    parser.add_argument("benchmarks", nargs="*", help="Specific benchmarks to build for (e.g. scicode colbench).")
    args = parser.parse_args()

    log("================================================================", Colors.CYAN)
    log("      COMPREHENSIVE DOCKER IMAGE PREBUILD", Colors.CYAN)
    log("================================================================", Colors.CYAN)

    failures = 0
    
    # 1. Base runner
    if not build_base_runner(args.force): 
        failures += 1
    
    # 2. Agent environments (always needed)
    if not build_agent_envs(args.benchmarks, args.force): 
        failures += 1
    
    # 3. Benchmark specific images
    if not args.benchmarks or "scienceagentbench" in args.benchmarks:
        if not build_sab_base(args.force): 
            failures += 1
    
    if not args.benchmarks or "scicode" in args.benchmarks:
        if not build_scicode_eval(args.force): 
            failures += 1

    # 4. CoreBench data (if corebench requested or all)
    if not args.benchmarks or any(b in ["corebench", "corebench_hard"] for b in args.benchmarks):
        if not ensure_corebench_data():
            failures += 1

    if failures == 0:
        log("SUCCESS: All images and data ready!", Colors.GREEN)
        sys.exit(0)
    else:
        log(f"FAILED: {failures} step(s) failed", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()
