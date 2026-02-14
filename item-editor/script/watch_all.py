#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
import json
import re
from pathlib import Path
from hal_common import Colors, detect_data_root, get_run_root

def list_logs_roots():
    repo_root = Path(__file__).resolve().parents[1]
    run_root = get_run_root()
    roots = set()
    roots.add(repo_root / "result" / ".hal_data" / "logs")
    roots.add(run_root / "logs")
    roots.add(repo_root / "logs")
    return [r for r in roots if r.exists()]

def list_results_roots():
    repo_root = Path(__file__).resolve().parents[1]
    run_root = get_run_root()
    roots = set()
    roots.add(repo_root / "result" / ".hal_data" / "results")
    roots.add(run_root / "results")
    roots.add(repo_root / "results")
    return [r for r in roots if r.exists()]

def collect_logs(prefix):
    all_logs = []
    
    # 1. Benchmark run logs
    for root in list_logs_roots():
        for dir_path in root.glob("benchmark_run_*"):
            if not dir_path.is_dir(): continue
            config_file = dir_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                        if config.get("prefix") == prefix:
                            all_logs.extend(list(dir_path.glob("*.log")))
                except: pass
                
    # 2. Results verbose logs
    for root in list_results_roots():
        # Using find-like glob
        for log_file in root.rglob(f"*{prefix}*_verbose.log"):
            all_logs.append(log_file)
            
    return list(set(all_logs))

def colorize_line(line, prefix):
    # Simplified colorization logic
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Extract run ID for display
    display_run = ""
    # Try to find something that looks like a run ID in the path if we have it
    
    display_prefix = f"[{timestamp}] "
    
    line_lower = line.lower()
    if any(m in line_lower for m in ["results:", "accuracy", "score", "evaluation completed"]):
        return f"{Colors.BOLD}{Colors.GREEN}{display_prefix}{line}{Colors.NC}"
    elif any(m in line_lower for m in ["error", "exception", "failed", "traceback"]):
        return f"{Colors.RED}{display_prefix}{line}{Colors.NC}"
    elif any(m in line_lower for m in ["401", "403", "429", "500", "502", "503", "504", "timeout", "unauthorized"]):
        return f"{Colors.MAGENTA}{display_prefix}{line}{Colors.NC}"
    elif any(m in line_lower for m in ["success", "completed", "finished"]):
        return f"{Colors.GREEN}{display_prefix}{line}{Colors.NC}"
    elif any(m in line_lower for m in ["warning", "warn"]):
        return f"{Colors.YELLOW}{display_prefix}{line}{Colors.NC}"
    elif any(m in line_lower for m in ["starting", "running", "task"]):
        return f"{Colors.BLUE}{display_prefix}{line}{Colors.NC}"
    
    return f"{display_prefix}{line}"

import datetime

def main():
    parser = argparse.ArgumentParser(description="Watch logs for a specific run prefix.")
    parser.add_argument("--prefix", required=True, help="Prefix to watch.")
    args = parser.parse_args()
    
    print(f"{Colors.CYAN}============================================================{Colors.NC}")
    print(f"{Colors.CYAN}           LOG VIEWER MODE (Prefix: {args.prefix}){Colors.NC}")
    print(f"{Colors.CYAN}============================================================{Colors.NC}")
    print(f"{Colors.CYAN}Press Ctrl+C to stop{Colors.NC}")
    
    watched_files = set()
    processes = {}

    try:
        while True:
            current_logs = collect_logs(args.prefix)
            new_files = [f for f in current_logs if f not in watched_files]
            
            if new_files:
                for f in new_files:
                    print(f"{Colors.CYAN}New log detected: {f.name}{Colors.NC}")
                    # Start tailing new file
                    proc = subprocess.Popen(["tail", "-f", "-n", "50", str(f)], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT,
                                         text=True)
                    processes[f] = proc
                    watched_files.add(f)
            
            # Check for output from all processes
            for f, proc in list(processes.items()):
                # Non-blocking read of all available lines
                while True:
                    import selectors
                    sel = selectors.DefaultSelector()
                    sel.register(proc.stdout, selectors.EVENT_READ)
                    events = sel.select(timeout=0.01) # Very short timeout
                    if not events:
                        break
                    
                    line = proc.stdout.readline()
                    if line:
                        print(colorize_line(line.strip(), args.prefix))
                    else:
                        break
                
                # Check if process is still alive
                if proc.poll() is not None:
                    del processes[f]
                    watched_files.remove(f)

            if not watched_files:
                print(f"{Colors.YELLOW}No log files found for prefix '{args.prefix}'. Waiting...{Colors.NC}")
                time.sleep(5)
            else:
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}Stopping...{Colors.NC}")
        for proc in processes.values():
            proc.terminate()

if __name__ == "__main__":
    main()
