#!/usr/bin/env python3
"""
Unified HAL Cleanup Tool

Handles:
1. Killing hung benchmark processes
2. Stopping/removing benchmark Docker containers
3. Cleaning up Docker images and cache
4. Reporting disk usage
"""

import os
import sys
import subprocess
import argparse
import datetime
import shutil
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "script" / "utils"))
from hal_common import log, Colors

# =============================================================================
# Helper Utilities
# =============================================================================

def run_command(cmd, shell=False, capture_output=True):
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=capture_output, text=True)
        return result.stdout.strip()
    except Exception:
        return ""

def get_disk_usage(path="/"):
    return run_command(["df", "-h", path]).split("
")[-1]

# =============================================================================
# Process Killing
# =============================================================================

def kill_processes():
    log("Step 1: Killing HAL Evaluation Processes...", Colors.BLUE)
    
    patterns = [
        # Main runners
        "run_all_benchmarks.sh",
        "run_benchmark_with_data.sh",
        "tail -f.*log",
        # Python scripts
        "eval_rubric.py",
        "fixing_pipeline.py",
        "claude_fixer",
        "scripts/run_.*_fixes.py",
        "run_.*_fixes.py",
        "judge.py",
        "pipeline.py",
        "hal.cli",
        "hal/cli.py",
        "python -m hal.cli",
        # Agent processes
        "run_agent.py",
        "scicode_tool_calling_agent",
        "hal_generalist_agent",
        "SWE-agent",
        "assistantbench_browser_agent",
        "colbench_example_agent",
        "smolagents",
        # Generic python scripts in our repo
        "python.*-u scripts/",
        # Final sweep
        "hal-eval",
        "hal_eval"
    ]
    
    for pattern in patterns:
        subprocess.run(["pkill", "-9", "-f", pattern], stderr=subprocess.DEVNULL)
    
    log("  [DONE] Processes signals sent", Colors.GREEN)

# =============================================================================
# Docker Cleanup
# =============================================================================

def cleanup_docker(aggressive=False, images=False):
    log("Step 2: Cleaning up Docker containers...", Colors.BLUE)
    
    if not shutil.which("docker"):
        log("  [SKIP] Docker CLI not found", Colors.YELLOW)
        return

    # 1. Kill benchmark containers
    filters = ["name=agentrun", "name=agentpool", "name=agent-env", "name=agentpreflight", "name=hal", "name=benchmark"]
    found_any = False
    for f in filters:
        ids = run_command(["docker", "ps", "-q", "--filter", f])
        if ids:
            found_any = True
            for cid in ids.split():
                subprocess.run(["docker", "kill", cid], stderr=subprocess.DEVNULL)
    
    if aggressive:
        log("  Aggressive mode: Killing ALL containers...", Colors.YELLOW)
        all_ids = run_command(["docker", "ps", "-q"])
        if all_ids:
            for cid in all_ids.split():
                subprocess.run(["docker", "kill", cid], stderr=subprocess.DEVNULL)

    # 2. Remove stopped containers
    log("  Removing stopped containers...", Colors.BLUE)
    if aggressive:
        ids = run_command(["docker", "ps", "-aq"])
    else:
        # Remove all exited/created containers
        ids = run_command(["docker", "ps", "-aq", "--filter", "status=exited", "--filter", "status=created"])
    
    if ids:
        for cid in ids.split():
            subprocess.run(["docker", "rm", "-f", cid], stderr=subprocess.DEVNULL)

    # 3. Images cleanup (if requested)
    if images:
        log("Step 3: Cleaning up Docker images...", Colors.BLUE)
        
        # Remove dangling images
        subprocess.run(["docker", "image", "prune", "-f"], stderr=subprocess.DEVNULL)
        
        # Remove agent-env-* images (they are high volume)
        log("  Removing ephemeral agent-env images...", Colors.BLUE)
        image_data = run_command(["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"])
        if image_data:
            for line in image_data.split("
"):
                if "agent-env-" in line and "hal-agent-runner" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        subprocess.run(["docker", "rmi", "-f", parts[1]], stderr=subprocess.DEVNULL)
        
        # Prune build cache
        log("  Pruning build cache...", Colors.BLUE)
        subprocess.run(["docker", "builder", "prune", "-af"], stderr=subprocess.DEVNULL)

    log("  [DONE] Docker cleanup finished", Colors.GREEN)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified HAL Cleanup Tool")
    parser.add_argument("--aggressive", action="store_true", help="Kill and remove ALL Docker containers.")
    parser.add_argument("--images", action="store_true", help="Also clean up ephemeral images and build cache.")
    parser.add_argument("--only-docker", action="store_true", help="Only perform Docker cleanup, don't kill local processes.")
    parser.add_argument("--only-processes", action="store_true", help="Only kill processes, don't touch Docker.")
    args = parser.parse_args()
    
    log("============================================================", Colors.CYAN)
    log("                HAL UNIFIED CLEANUP TOOL", Colors.CYAN)
    log("============================================================", Colors.CYAN)
    
    initial_usage = get_disk_usage()
    log(f"Initial disk usage: {initial_usage}", Colors.BLUE)
    print()

    if not args.only_docker:
        kill_processes()
        print()

    if not args.only_processes:
        cleanup_docker(args.aggressive, args.images)
        print()
    
    # Final Report
    log("============================================================", Colors.CYAN)
    final_usage = get_disk_usage()
    log(f"Final disk usage:   {final_usage}", Colors.GREEN)
    
    log("Remaining Docker containers:", Colors.BLUE)
    remaining = run_command(["docker", "ps", "--format", "table {{.ID}}	{{.Names}}	{{.Status}}"])
    if remaining and "ID" in remaining:
        print(remaining)
    else:
        log("  None.", Colors.NC)
        
    log("============================================================", Colors.CYAN)

if __name__ == "__main__":
    main()
