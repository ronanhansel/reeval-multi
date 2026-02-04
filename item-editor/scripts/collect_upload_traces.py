#!/usr/bin/env python3
"""
Collect all UPLOAD.json files from result directories matching a prefix.
"""

from __future__ import annotations

import argparse
import os
import shutil
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
HAL_HARNESS_PATH = REPO_ROOT / "hal-harness"
if str(HAL_HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HAL_HARNESS_PATH))

BENCHMARKS = ("scicode", "scienceagentbench", "corebench", "colbench")
HAL_BENCHMARK_MAP = {
    "scicode": "scicode",
    "scienceagentbench": "scienceagentbench",
    "corebench": "corebench_hard",
    "colbench": "colbench_backend_programming",
}

def detect_run_root(script_dir: Path) -> Path:
    # 1. If results/logs/etc exist in CWD, use it
    cwd = Path(".").resolve()
    for d in ["results", "logs", ".hal_data", ".hal-data", "output"]:
        if (cwd / d).exists():
            return cwd

    script_dir = script_dir.resolve()
    if script_dir.name == "scripts":
        repo_root = script_dir.parent
        project_name = repo_root.name
    else:
        repo_root = script_dir
        project_name = script_dir.name

    # 2. If results/logs exist in repo_root, use it
    for d in ["results", "logs", ".hal_data", ".hal-data", "output"]:
        if (repo_root / d).exists():
            return repo_root

    # 3. Check home directory for a data root
    home = Path.home()
    for d in [".hal_data", ".hal-data"]:
        root = home / d / "hal_runs" / os.getlogin() / project_name
        if root.is_dir():
            return root

    # 4. DATA_PATH fallback
    data_path = os.environ.get("DATA_PATH") or os.environ.get("HAL_DATA_ROOT")
    if data_path and Path(data_path).is_dir():
        namespace = os.environ.get("HAL_DATA_NAMESPACE") or os.environ.get("USER") or os.getlogin() or "user"
        root = Path(data_path) / "hal_runs" / namespace / project_name
        if root.is_dir():
            return root
            
    return repo_root

def find_run_ids_from_results(
    run_root: Path,
    repo_root: Path,
    benchmark: str,
    prefix_pattern: str,
) -> List[Tuple[str, str]]:

    hal_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)

    candidates = [
        run_root / "results" / hal_name,
        run_root / "results" / benchmark,
        run_root / ".hal_data" / "results" / hal_name,
        run_root / ".hal-data" / "results" / hal_name,
        repo_root / "results" / hal_name,
        repo_root / "results" / benchmark,
    ]

    run_ids: List[Tuple[str, str]] = []

    try:
        regex = re.compile(prefix_pattern)
    except re.error:
        regex = None

    for base in candidates:
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if regex:
                m = regex.search(child.name)
                if m:
                    run_ids.append((child.name, m.group(0)))
            elif prefix_pattern in child.name:
                run_ids.append((child.name, prefix_pattern))
    return sorted(list(set(run_ids)))

def resolve_run_dir(run_root: Path, repo_root: Path, benchmark: str, run_id: str) -> Optional[Path]:

    hal_name = HAL_BENCHMARK_MAP.get(benchmark, benchmark)

    candidates = [
        run_root / "results" / hal_name / run_id,
        run_root / "results" / benchmark / run_id,
        run_root / ".hal_data" / "results" / hal_name / run_id,
        run_root / ".hal-data" / "results" / hal_name / run_id,
        repo_root / "results" / hal_name / run_id,
        repo_root / "results" / benchmark / run_id,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None

def parse_run_id(rid: str) -> Tuple[str, str]:
    """
    Split a run ID into (base, timestamp).
    HAL run IDs typically end with _YYYYMMDD_HHMMSS.
    """
    parts = rid.split("_")
    if len(parts) >= 2:
        # Check for _YYYYMMDD_HHMMSS at the end
        date_part = parts[-2]
        time_part = parts[-1]
        if len(date_part) == 8 and date_part.isdigit() and len(time_part) == 6 and time_part.isdigit():
            return "_".join(parts[:-2]), f"{date_part}_{time_part}"
    return rid, ""

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all UPLOAD.json files from result directories matching a prefix.")
    parser.add_argument("--prefix", required=True, help="Prefix pattern (string or regex) like beach[0-9]+_")
    parser.add_argument("--output", required=True, help="Directory to save collected UPLOAD.json files")
    parser.add_argument("--run-root", help="Override run root path")
    parser.add_argument("--benchmark", action="append", help="Select specific benchmark(s) to process.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    repo_root = script_dir
    run_root = Path(args.run_root) if args.run_root else detect_run_root(script_dir)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        active_benchmarks = [b for b in BENCHMARKS if b in args.benchmark]
    else:
        active_benchmarks = list(BENCHMARKS)

    # benchmark -> base_rid -> (timestamp, rid, upload_path, raw_path, eval_path)
    latest_runs: Dict[str, Dict[str, Tuple[str, str, Path, Optional[Path], Optional[Path]]]] = {}
    
    for benchmark in active_benchmarks:
        latest_runs[benchmark] = {}
        pairs = find_run_ids_from_results(run_root, repo_root, benchmark, args.prefix)
        
        for rid, actual_pfx in pairs:
            run_dir = resolve_run_dir(run_root, repo_root, benchmark, rid)
            if not run_dir:
                continue

            upload_path = run_dir / f"{rid}_UPLOAD.json"
            raw_path = run_dir / f"{rid}_RAW_SUBMISSIONS.jsonl"
            eval_path = run_dir / f"{rid}_eval.jsonl"
            
            if upload_path.exists():
                base, timestamp = parse_run_id(rid)
                current_raw = raw_path if raw_path.exists() else None
                current_eval = eval_path if eval_path.exists() else None
                
                if base not in latest_runs[benchmark] or timestamp > latest_runs[benchmark][base][0]:
                    latest_runs[benchmark][base] = (timestamp, rid, upload_path, current_raw, current_eval)

    count_upload = 0
    count_raw = 0
    count_eval = 0
    
    traces_out_dir = output_dir / "traces"
    raw_out_dir = output_dir / "raw_submission"
    
    traces_out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    for benchmark, bases in latest_runs.items():
        for base, (timestamp, rid, upload_path, raw_path, eval_path) in bases.items():
            # Handle UPLOAD.json
            dest_name = f"{benchmark}_{rid}_UPLOAD.json"
            dest_path = traces_out_dir / dest_name
            
            for existing_file in traces_out_dir.glob(f"{benchmark}_{base}_*_UPLOAD.json"):
                if existing_file.name != dest_name:
                    print(f"Removing old UPLOAD version in output: {existing_file.name}")
                    existing_file.unlink()

            if not dest_path.exists():
                print(f"Copying {upload_path} to {dest_path}")
                shutil.copy2(upload_path, dest_path)
                count_upload += 1

            # Handle RAW_SUBMISSIONS.jsonl
            if raw_path:
                raw_dest_name = f"{benchmark}_{rid}_RAW_SUBMISSIONS.jsonl"
                raw_dest_path = raw_out_dir / raw_dest_name
                
                for existing_file in raw_out_dir.glob(f"{benchmark}_{base}_*_RAW_SUBMISSIONS.jsonl"):
                    if existing_file.name != raw_dest_name:
                        print(f"Removing old RAW version in output: {existing_file.name}")
                        existing_file.unlink()

                if not raw_dest_path.exists():
                    print(f"Copying {raw_path} to {raw_dest_path}")
                    shutil.copy2(raw_path, raw_dest_path)
                    count_raw += 1

            # Handle eval.jsonl (ScienceAgentBench)
            if eval_path:
                eval_dest_name = f"{benchmark}_{rid}_eval.jsonl"
                eval_dest_path = raw_out_dir / eval_dest_name
                
                for existing_file in raw_out_dir.glob(f"{benchmark}_{base}_*_eval.jsonl"):
                    if existing_file.name != eval_dest_name:
                        print(f"Removing old EVAL version in output: {existing_file.name}")
                        existing_file.unlink()

                if not eval_dest_path.exists():
                    print(f"Copying {eval_path} to {eval_dest_path}")
                    shutil.copy2(eval_path, eval_dest_path)
                    count_eval += 1
    
    print(f"\nCollected {count_upload} latest UPLOAD.json files to {traces_out_dir}")
    print(f"Collected {count_raw} latest RAW_SUBMISSIONS.jsonl files to {raw_out_dir}")
    if count_eval > 0:
        print(f"Collected {count_eval} latest eval.jsonl files to {raw_out_dir}")

if __name__ == "__main__":
    main()
