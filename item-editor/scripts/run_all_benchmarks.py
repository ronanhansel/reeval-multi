#!/usr/bin/env python3
import os
import sys
import argparse
import json
import subprocess
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from hal_common import log, Colors, REPO_ROOT, setup_data_dirs, get_dotenv_path

# Default benchmarks
ALL_BENCHMARKS = ["scicode", "scienceagentbench", "corebench", "colbench"]

def increment_prefix(prefix: str) -> str:
    import re
    match = re.search(r'([^0-9]*)([0-9]+)([^0-9]*)', prefix)
    if match:
        base, num, suffix = match.groups()
        next_num = int(num) + 1
        return f"{base}{next_num}{suffix}"
    return f"{prefix}1"

def get_prefix_num(prefix: str) -> int:
    import re
    match = re.search(r'([0-9]+)', prefix)
    if match:
        return int(match.group(1))
    return -1

def run_benchmark(benchmark, prefix, args, log_dir):
    log_file = log_dir / f"{benchmark}.log"
    exit_code_file = log_dir / f"{benchmark}.exit_code"
    
    benchmark_prefix = f"{benchmark}_{prefix}"
    
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_benchmark_fixes.py"),
        "--benchmark", benchmark,
        "--all-configs",
        "--prefix", benchmark_prefix,
        "--docker",
        "--parallel-models", str(args.parallel_models),
        "--parallel-tasks", str(args.parallel_tasks),
        "--resume"
    ]
    
    if args.trace_mode: cmd.extend(["--trace-mode", args.trace_mode])
    if args.sample_tasks: cmd.extend(["--sample-tasks", str(args.sample_tasks)])
    if args.sample_seed: cmd.extend(["--sample-seed", str(args.sample_seed)])
    if args.timeout: cmd.extend(["--timeout", str(args.timeout)])
    
    if args.no_fix:
        cmd.append("--no-fix")
    elif not args.fix_only:
        cmd.append("--all-tasks")
    
    env = os.environ.copy()
    if args.skip_root_check:
        env["HAL_SKIP_ROOT_CHECK"] = "1"
    
    # Map colors
    colors = {
        "scicode": Colors.BLUE,
        "scienceagentbench": Colors.GREEN,
        "corebench": Colors.YELLOW,
        "colbench": Colors.CYAN
    }
    color = colors.get(benchmark, Colors.WHITE)

    log(f"Starting {benchmark}...", color)
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        
        # Tailing thread
        def tail_log():
            try:
                with open(log_file, "r") as tf:
                    tf.seek(0, 2) # Go to end
                    while process.poll() is None:
                        line = tf.readline()
                        if line:
                            # Filter and colorize
                            l = line.strip()
                            if any(m in l.lower() for m in ["error", "exception", "failed", "traceback", "401"]):
                                print(f"{Colors.RED}[{benchmark}] {l}{Colors.NC}", flush=True)
                            elif any(m in l for m in ["COMPLETED", "FINISHED", "All done"]):
                                print(f"{Colors.GREEN}[{benchmark}] {l}{Colors.NC}", flush=True)
                            elif "STEP:" in l:
                                print(f"{color}[{benchmark}] {l}{Colors.NC}", flush=True)
                        else:
                            time.sleep(0.5)
            except Exception as e:
                print(f"Tail error for {benchmark}: {e}")

        threading.Thread(target=tail_log, daemon=True).start()
        
        return process

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Runner")
    parser.add_argument("--prefix", default="moon1_")
    parser.add_argument("--benchmarks", help="Comma-separated list of benchmarks")
    parser.add_argument("--parallel", type=int, help="Total parallel tasks (caps at 800)")
    parser.add_argument("--parallel-models", type=int, default=10)
    parser.add_argument("--parallel-tasks", type=int, default=10)
    parser.add_argument("--trace-mode", help="Set HAL_TRACE_MODE")
    parser.add_argument("--sample-tasks", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--fix-only", action="store_true")
    parser.add_argument("--no-fix", action="store_true")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--until", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--skip-root-check", action="store_true")
    parser.add_argument("--force-reset", action="store_true")

    args = parser.parse_args()
    
    # Setup environment and directories
    data_dirs = setup_data_dirs()
    logs_base = Path(os.environ["HAL_LOGS_DIR"])

    # Benchmarks to run
    if args.benchmarks:
        active_benchmarks = args.benchmarks.split(",")
    else:
        active_benchmarks = ALL_BENCHMARKS

    # Phase 1: Pre-build images and data
    log("PHASE 1: PRE-BUILDING DOCKER IMAGES & DATA", Colors.CYAN)
    prebuild_script = REPO_ROOT / "scripts" / "prebuild_all_images.py"
    subprocess.run([sys.executable, str(prebuild_script)] + active_benchmarks, check=True)

    # Phase 2: Task Counts
    log("PHASE 2: TASK COUNT SUMMARY", Colors.CYAN)
    from run_benchmark_fixes import load_benchmark_dataset, get_task_ids_with_fixes
    
    total_tasks = 0
    task_results = []
    for b in active_benchmarks:
        if args.no_fix or not args.fix_only:
            dataset = load_benchmark_dataset(b)
            count = len(dataset) if dataset else 0
        else:
            count = len(get_task_ids_with_fixes(b))
        task_results.append((b, count))
        total_tasks += count
    
    mode_str = "fixes only" if args.fix_only and not args.no_fix else "all tasks"
    print(f"\nMode: {mode_str} ({total_tasks} tasks total)")
    for b, count in task_results:
        print(f"  - {b}: {count} tasks")
    print()

    prefix = args.prefix
    repeat_count = args.repeat

    while True:
        log(f"      COMPREHENSIVE BENCHMARK RUNNER (Prefix: {prefix})", Colors.CYAN)
        log("============================================================", Colors.CYAN)
        
        bench_key = "+".join(active_benchmarks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = logs_base / f"benchmark_run_{prefix}__{bench_key}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log(f"Log Directory: {log_dir}", Colors.BLUE)
        
        # Save config
        with open(log_dir / "config.json", "w") as f:
            json.dump({
                "prefix": prefix,
                "parallel_models": args.parallel_models,
                "parallel_tasks": args.parallel_tasks,
                "benchmarks": active_benchmarks
            }, f, indent=4)

        processes = []
        for benchmark in active_benchmarks:
            p = run_benchmark(benchmark, prefix, args, log_dir)
            processes.append(p)
            time.sleep(2)

        # Wait for all
        for p in processes:
            p.wait()

        # Check for loop
        should_loop = False
        if repeat_count > 0:
            repeat_count -= 1
            should_loop = True
        elif args.until:
            current_num = get_prefix_num(prefix)
            if current_num >= 0 and current_num < args.until:
                should_loop = True
        
        if should_loop:
            prefix = increment_prefix(prefix)
            log(f"Incrementing prefix to {prefix}", Colors.YELLOW)
        else:
            break

    log("All iterations complete.", Colors.GREEN)

if __name__ == "__main__":
    main()
