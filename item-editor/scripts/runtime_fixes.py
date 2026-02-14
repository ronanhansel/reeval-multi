#!/usr/bin/env python3
"""
Unified HAL Benchmark Execution Engine

Consolidates run_benchmark_fixes.py and run_all_benchmarks.py into one tool.
Supports single-config runs, bulk benchmark runs, and automated loops.

Features:
- Parallel model execution
- Parallel task execution
- Parallel benchmark execution (in bulk mode)
- Fix application (prompt/env patching)
- Real-time colorized log tailing
- Automated prefix loops
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration & Paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
HAL_HARNESS = REPO_ROOT / "hal-harness"

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def log(msg: str, prefix: str = "main", color: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{ts}] [{prefix}] {msg}{Colors.NC}", flush=True)

# Load environment
from dotenv import load_dotenv
if not os.environ.get("HAL_DOTENV_PATH"):
    default_dotenv = HAL_HARNESS / ".env"
    if default_dotenv.exists():
        os.environ["HAL_DOTENV_PATH"] = str(default_dotenv)
        load_dotenv(default_dotenv, override=False)
load_dotenv(override=False)

def _resolve_data_dir(env_key: str, default_path: Path) -> Path:
    raw = os.environ.get(env_key)
    path = Path(raw) if raw else default_path
    if not path.is_absolute(): path = REPO_ROOT / path
    return path

FIXES_DIR = REPO_ROOT / "fixes"
TRACES_DIR = _resolve_data_dir("HAL_TRACES_DIR", REPO_ROOT / "traces")
RESULTS_DIR = _resolve_data_dir("HAL_RESULTS_DIR", REPO_ROOT / "results")
TMP_DIR = _resolve_data_dir("HAL_TMP_DIR", REPO_ROOT / ".tmp")
LOGS_BASE = _resolve_data_dir("HAL_LOGS_DIR", REPO_ROOT / "logs")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Mappings
BENCHMARK_HAL_NAME_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "corebench_hard": "corebench_hard",
    "usaco": "usaco",
    "colbench": "colbench_backend_programming",
}

BENCHMARK_DATASET_ENV_VAR = {
    "scicode": "SCICODE_DATASET_PATH",
    "scienceagentbench": "SCIENCEAGENTBENCH_DATASET_PATH",
    "corebench": "HAL_COREBENCH_DATASET_PATH",
    "colbench": "COLBENCH_BACKEND_DATASET_PATH",
}

BENCHMARK_TASK_ID_FIELD = {
    "scicode": "problem_id",
    "scienceagentbench": "instance_id",
    "corebench": "capsule_id",
    "colbench": "id",
    "usaco": "problem_id",
}

DEFAULT_BENCHMARKS = ["scicode", "scienceagentbench", "corebench", "colbench"]

# =============================================================================
# Helper Logic
# =============================================================================

def get_model_quirks(model_id: str) -> Dict[str, bool]:
    # Check if we can import from shared
    try:
        from shared.model_utils import supports_temperature, supports_reasoning_effort
        return {
            'supports_temperature': supports_temperature(model_id),
            'supports_reasoning_effort': supports_reasoning_effort(model_id),
        }
    except:
        m = model_id.lower()
        is_reasoning = any(p in m for p in ['o1', 'o3', 'o4', 'gpt-5'])
        return {
            'supports_temperature': not is_reasoning,
            'supports_reasoning_effort': is_reasoning,
        }

def build_agent_args(entry: Dict[str, Any]) -> Dict[str, Any]:
    mid = entry.get("model_id", "")
    q = get_model_quirks(mid)
    args = {"model_name": mid}
    if "temperature" in entry and q['supports_temperature']: args["temperature"] = entry["temperature"]
    if "reasoning_effort" in entry and q['supports_reasoning_effort']: args["reasoning_effort"] = entry["reasoning_effort"]
    for k in ["max_steps", "budget", "max_tokens", "use_self_debug", "use_knowledge"]:
        if k in entry: args[k] = entry[k]
    return args

def tail_log_to_stdout(benchmark: str, log_path: Path, color: str):
    """Thread function to tail a log file to stdout with color."""
    try:
        # Wait for file to exist
        for _ in range(20):
            if log_path.exists(): break
            time.sleep(0.5)
        
        if not log_path.exists():
            return

        with open(log_path, "r") as f:
            f.seek(0, 2) # Go to end
            while True:
                line = f.readline()
                if line:
                    l = line.strip()
                    if not l: continue
                    # Colorize based on content
                    if any(m in l.lower() for m in ["error", "exception", "failed", "traceback", "401"]):
                        print(f"{Colors.RED}[{benchmark}] {l}{Colors.NC}", flush=True)
                    elif any(m in l for m in ["COMPLETED", "FINISHED", "All done"]):
                        print(f"{Colors.GREEN}[{benchmark}] {l}{Colors.NC}", flush=True)
                    elif "STEP:" in l or "Job started:" in l:
                        print(f"{color}[{benchmark}] {l}{Colors.NC}", flush=True)
                    else:
                        print(f"[{benchmark}] {l}", flush=True)
                else:
                    if not threading.main_thread().is_alive(): break
                    time.sleep(0.5)
    except Exception as e:
        pass

# =============================================================================
# Execution logic
# =============================================================================

def run_hal_job(benchmark: str, config_key: str, entry: Dict[str, Any], prefix: str, dataset_path: Path, args: argparse.Namespace, log_file: Optional[Path] = None) -> bool:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}{config_key}_{ts}"
    
    hal_cmd = [
        sys.executable, "-m", "hal.cli",
        "--benchmark", BENCHMARK_HAL_NAME_MAP.get(benchmark, benchmark),
        "--agent_name", f"{prefix}{config_key}",
        "--agent_function", entry.get("agent_function", "main.run"),
        "--agent_dir", str(REPO_ROOT / entry.get("agent_dir")),
        "--run_id", run_id,
        "--max_concurrent", str(args.parallel_tasks),
        "--continue_run", "--ignore_errors"
    ]
    if args.docker: hal_cmd.append("--docker")
    # Priority: args.max_tasks from CLI
    if args.max_tasks: hal_cmd.extend(["--max_tasks", str(args.max_tasks)])
    
    agent_args = build_agent_args(entry)
    for k, v in agent_args.items():
        hal_cmd.extend(["-A", f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}"])
    hal_cmd.extend(["-A", f"benchmark_name={benchmark}"])
    if args.timeout: hal_cmd.extend(["-A", f"timeout={args.timeout}"])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HAL_HARNESS}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["HAL_WEAVE_PROJECT"] = f"{prefix.rstrip('_')}_{benchmark}"
    if args.trace_mode: env["HAL_TRACE_MODE"] = args.trace_mode
    
    env_var = BENCHMARK_DATASET_ENV_VAR.get(benchmark)
    if env_var: env[env_var] = str(dataset_path)

    try:
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"\n--- JOB START: {config_key} ---\n")
                f.flush()
                proc = subprocess.run(hal_cmd, env=env, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.run(hal_cmd, env=env, cwd=REPO_ROOT)
            
        # Move trace
        src = RESULTS_DIR / benchmark / run_id / f"{run_id}_UPLOAD.json"
        if src.exists():
            dest = TRACES_DIR / f"{prefix}{src.name}"
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            
        return proc.returncode == 0
    except Exception as e:
        log(f"Job failed: {e}", benchmark, Colors.RED)
        return False

def load_benchmark_dataset(benchmark: str):
    try:
        if benchmark == "scicode":
            from datasets import load_dataset
            return list(load_dataset("SciCode1/SciCode", split="test"))
        elif benchmark == "scienceagentbench":
            from datasets import load_dataset
            return list(load_dataset("osunlp/ScienceAgentBench", split="validation"))
        elif benchmark in ("corebench", "corebench_hard"):
            p = HAL_HARNESS / "hal/benchmarks/corebench/core_test.json"
            return json.loads(p.read_text())
        elif benchmark == "colbench":
            p = HAL_HARNESS / "hal/benchmarks/colbench/data/backend_test.jsonl"
            tasks = []
            with open(p, "r") as f:
                for i, line in enumerate(f):
                    t = json.loads(line); t["id"] = str(i); tasks.append(t)
            return tasks
    except: return []

def run_suite(benchmark: str, prefix: str, args: argparse.Namespace, log_file: Optional[Path] = None) -> bool:
    sys.path.insert(0, str(HAL_HARNESS))
    if benchmark in ("corebench", "corebench_hard"):
        json_path = HAL_HARNESS / "hal" / "benchmarks" / "corebench" / "core_test.json"
        if not json_path.exists():
            gpg = json_path.with_suffix(".json.gpg")
            if gpg.exists(): subprocess.run(["gpg", "--batch", "--yes", "--passphrase", "reproducibility", "-o", str(json_path), "--decrypt", str(gpg)], check=True)
        from hal.benchmarks.corebench import CoreBenchHard
        CoreBenchHard(agent_dir=".", config={})
    
    dataset = load_benchmark_dataset(benchmark)
    if not dataset:
        log(f"No tasks found for {benchmark}. Check dataset path.", benchmark, Colors.RED)
        return False
    
    fix_target = "corebench_hard" if benchmark == "corebench" else benchmark
    fix_root = FIXES_DIR / fix_target
    has_fixes = set([d.name for d in fix_root.iterdir() if d.is_dir()]) if fix_root.exists() else set()
    id_field = BENCHMARK_TASK_ID_FIELD.get(benchmark, "id")
    
    final_ds = []
    fixed_count = 0
    for task in dataset:
        tid = str(task.get(id_field, ""))
        if not args.no_fix and tid in has_fixes:
            import copy
            mod = copy.deepcopy(task)
            fix_path = fix_root / tid
            for override in ["instruction_override", "env_override", "evaluation_override"]:
                p = fix_path / f"{override}.json"
                if p.exists():
                    data = json.loads(p.read_text())
                    mod[f"_fix_{override.replace('_override', '')}"] = data
                    if override == "instruction_override" and "clarifications" in data:
                        for f in ["instruction", "task_inst", "problem_statement"]:
                            if f in mod: mod[f] = str(mod[f]) + "\n\nCLARIFICATIONS:\n" + "\n".join(f"- {c}" for c in data["clarifications"]); break
            final_ds.append(mod)
            fixed_count += 1
        elif not args.fix_only:
            final_ds.append(task)
            
    if args.sample_tasks: final_ds = random.sample(final_ds, min(len(final_ds), args.sample_tasks))
    if not final_ds:
        log(f"Empty dataset for {benchmark} (fix-only={args.fix_only})", benchmark, Colors.YELLOW)
        return True

    log(f"Dataset: {len(final_ds)} tasks ({fixed_count} with fixes)", benchmark, Colors.GREEN)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ext = "jsonl" if benchmark == "colbench" else "json"
    ds_path = TMP_DIR / f"{benchmark}_run_{ts}.{ext}"
    if ext == "jsonl":
        with ds_path.open("w") as f:
            for t in final_ds: f.write(json.dumps(t) + "\n")
    else: ds_path.write_text(json.dumps(final_ds, indent=2))

    cfg_path = REPO_ROOT / "model_configs" / f"model_to_baseline_{benchmark}.json"
    if not cfg_path.exists():
        log(f"Config not found: {cfg_path}", benchmark, Colors.RED)
        return False
        
    selected = {k: v for k, v in json.loads(cfg_path.read_text()).items() if not k.startswith("_")}
    if args.configs: selected = {k: v for k, v in selected.items() if k in args.configs}
    if args.model_filter: selected = {k: v for k, v in selected.items() if args.model_filter.lower() in k.lower()}
    
    if not selected:
        log(f"No configs for {benchmark}", benchmark, Colors.YELLOW)
        return True

    if log_file:
        with open(log_file, "a") as f:
            f.write(f"Benchmark: {benchmark}\nTasks: {len(final_ds)}\nConfigs: {len(selected)}\n\n")

    log(f"Queuing {len(selected)} model configurations...", benchmark, Colors.CYAN)
    
    results = []
    if args.parallel_models > 1:
        with ThreadPoolExecutor(max_workers=args.parallel_models) as ex:
            futures = {ex.submit(run_hal_job, benchmark, k, v, prefix, ds_path, args, log_file): k for k, v in selected.items()}
            for f in as_completed(futures): results.append(f.result())
    else:
        for k, v in selected.items(): results.append(run_hal_job(benchmark, k, v, prefix, ds_path, args, log_file))
            
    ds_path.unlink(missing_ok=True)
    return all(results)

# =============================================================================
# Main
# =============================================================================

def increment_prefix(prefix: str) -> str:
    m = re.search(r'([^0-9]*)([0-9]+)([^0-9]*)', prefix)
    if m:
        base, num, suffix = m.groups()
        return f"{base}{int(num) + 1}{suffix}"
    return f"{prefix}1"

def main():
    parser = argparse.ArgumentParser(description="Unified HAL Benchmark Runner")
    # Selection
    parser.add_argument("--benchmark", "-b", help="Single benchmark to run.")
    parser.add_argument("--benchmarks", help="Comma-separated benchmarks for bulk run.")
    parser.add_argument("--config", "-c", dest="configs", action="append", help="Specific config key(s).")
    parser.add_argument("--model-filter", "-m", help="Filter configs by pattern.")
    # Parallelism
    parser.add_argument("--parallel-models", type=int, default=10, help="Concurrent configs (default: 10).")
    parser.add_argument("--parallel-tasks", type=int, default=10, help="Concurrent tasks per model (default: 10).")
    # Execution
    parser.add_argument("--prefix", default="moon1_", help="Output prefix (default: moon1_).")
    parser.add_argument("--docker", action="store_true", help="Use Docker isolation.")
    parser.add_argument("--trace-mode", help="Set HAL_TRACE_MODE (local, weave, etc.).")
    parser.add_argument("--fix-only", action="store_true", help="Only run tasks with fixes.")
    parser.add_argument("--no-fix", action="store_true", help="Baseline mode: ignore all fixes.")
    parser.add_argument("--timeout", type=int, help="Per-task timeout.")
    parser.add_argument("--max-tasks", type=int, help="Limit tasks per config.")
    parser.add_argument("--sample-tasks", type=int, help="Sample N random tasks.")
    parser.add_argument("--sample-seed", type=int, help="Seed for sampling.")
    # Loop
    parser.add_argument("--until", type=int, help="Increment prefix number until reaching N.")
    parser.add_argument("--repeat", type=int, default=0, help="Repeat loop N times.")
    # Misc
    parser.add_argument("--skip-root-check", action="store_true", help="Set HAL_SKIP_ROOT_CHECK=1.")
    parser.add_argument("--force-reset", action="store_true", help="Set HAL_FORCE_RESET=1.")

    args = parser.parse_args()
    if args.skip_root_check: os.environ["HAL_SKIP_ROOT_CHECK"] = "1"
    if args.force_reset: os.environ["HAL_FORCE_RESET"] = "1"

    active_benchmarks = args.benchmarks.split(",") if args.benchmarks else ([args.benchmark] if args.benchmark else DEFAULT_BENCHMARKS)
    prefix = args.prefix
    repeat_count = args.repeat

    while True:
        log(f"--- SUITE START: {prefix} ---", "suite", Colors.MAGENTA)
        
        bench_key = "+".join(active_benchmarks)
        ts_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = LOGS_BASE / f"benchmark_run_{prefix}__{bench_key}_{ts_dir}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log(f"Logs: {log_dir}", "suite", Colors.BLUE)

        with open(log_dir / "config.json", "w") as f:
            json.dump({"prefix": prefix, "parallel_models": args.parallel_models, "parallel_tasks": args.parallel_tasks, "benchmarks": active_benchmarks}, f, indent=4)

        colors = [Colors.BLUE, Colors.GREEN, Colors.YELLOW, Colors.CYAN, Colors.MAGENTA]
        threads = []
        
        for i, bench in enumerate(active_benchmarks):
            log_path = log_dir / f"{bench}.log"
            color = colors[i % len(colors)]
            
            # Start suite thread
            t = threading.Thread(target=run_suite, args=(bench, prefix, args, log_path), daemon=True)
            t.start()
            threads.append(t)
            
            # Start tailing thread
            tail_t = threading.Thread(target=tail_log_to_stdout, args=(bench, log_path, color), daemon=True)
            tail_t.start()
            
            time.sleep(2) 

        # Wait for all benchmark suites to finish
        for t in threads:
            t.join()

        # Check loop
        should_loop = False
        if repeat_count > 0:
            repeat_count -= 1; should_loop = True
        elif args.until:
            m = re.search(r'([0-9]+)', prefix)
            num = int(m.group(1)) if m else -1
            if 0 <= num < args.until: should_loop = True
        
        if should_loop:
            prefix = increment_prefix(prefix)
            log(f"Next prefix: {prefix}", "loop", Colors.YELLOW)
        else: break

    log("Execution suite complete.", "main", Colors.GREEN)

if __name__ == "__main__":
    main()
