#!/usr/bin/env python3
import docker
import os
import sys
import toml
from pathlib import Path

def build_scicode_image():
    client = docker.from_env(timeout=600)
    
    # Locate SciCode pyproject.toml
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    pyproject_path = repo_root / "hal-harness" / "hal" / "benchmarks" / "SciCode" / "pyproject.toml"
    
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)
        
    print(f"Reading dependencies from {pyproject_path}")
    pyproject = toml.load(pyproject_path)
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    
    if not dependencies:
        print("Warning: No dependencies found in pyproject.toml")
    
    # dependencies usually look like ["numpy", "scipy", ...]
    # we join them for pip install
    pip_packages = " ".join(dependencies)
    print(f"Dependencies to install: {pip_packages}")
    
    dockerfile_content = f"""
FROM python:3.11

# Install system dependencies if any (e.g. for scipy/numpy compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install SciCode dependencies
RUN pip install --no-cache-dir {pip_packages}

# Create app directory
WORKDIR /app
"""
    
    print("Building scicode-eval:latest...")
    try:
        # Create a temporary directory context
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Dockerfile").write_text(dockerfile_content)
            
            # Use low-level API to avoid decode=True issues with high-level images.build
            response = client.api.build(
                path=tmpdir,
                tag="scicode-eval:latest",
                rm=True,
                decode=True
            )
            
            for log in response:
                if 'stream' in log:
                    print(log['stream'], end='')
                if 'error' in log:
                    raise Exception(log['error'])
                    
        # Verify image was created
        client.images.get("scicode-eval:latest")
        print("\nSUCCESS: Built scicode-eval:latest")
        return 0
        
    except docker.errors.BuildError as e:
        print(f"\nBUILD ERROR: {e}")
        for log in e.build_log:
            if 'stream' in log:
                print(log['stream'], end='')
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(build_scicode_image())
